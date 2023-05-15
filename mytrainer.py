from __future__ import absolute_import, division, print_function


import time
from tkinter import NO
from xml.etree.ElementTree import TreeBuilder
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

from networks.model.lite_mono import LiteMono
# torch.backends.cudnn.benchmark = True


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Trainer:
    def __init__(self, config):

        self.cfg = config

        self.log_path = os.path.join(self.cfg["log_dir"], self.cfg["name"])

        # checking height and width are multiples of 32
        assert self.cfg["dataset"]["height"] % 32 == 0, "'height' must be a multiple of 32"
        assert self.cfg["dataset"]["width"]% 32 == 0, "'width' must be a multiple of 32"

        self.height = self.cfg["dataset"]["height"]
        self.width = self.cfg["dataset"]["width"]

        self.min_depth = self.cfg["min_depth"]
        self.max_depth = self.cfg["max_depth"]

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = torch.device("cpu" if self.cfg["no_cuda"] else "cuda")
        self.profile = self.cfg["profile"]


        self.num_scales = len(self.cfg["train_params"]["scales"])
        self.frame_ids = self.cfg["train_params"]["frame_ids"]
        self.scales = self.cfg["train_params"]["scales"]

        self.num_pose_frames = 2 if self.cfg["ARCH"]["pose_model_input"] == "pairs" else self.num_input_frames

        assert self.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.cfg["train_params"]["use_stereo"] and self.frame_ids[0] == [0])

        
        if self.cfg["train_params"]["use_stereo"]:
            self.frame_ids.append("s")


        # self.models["encoder"] = networks.LiteMono(model=self.cfg["ARCH"]["model"],
        #                                            drop_path_rate=self.cfg["ARCH"]["drop_path"])

        # self.models["encoder"].to(self.device)
        # self.parameters_to_train += list(self.models["encoder"].parameters())

        # self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
        #                                              self.scales)
        # self.models["depth"].to(self.device)

        self.models["depth"] = LiteMono(drop_path_rate=self.cfg["ARCH"]["drop_path"])


        self.parameters_to_train += list(self.models["depth"].parameters())
        self.models["depth"].to(self.device)


        self.pose_model_type = None
        if self.use_pose_net:

            self.pose_model_type = self.cfg["ARCH"]["pose_model_type"]
            if self.pose_model_type == "separate_resnet":
                self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                    self.cfg["ARCH"]["num_layers"],
                    self.cfg["ARCH"]["weights_init"] == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models_pose["pose_encoder"].to(self.device)
                self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models_pose["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.pose_model_type == "posecnn":
                self.models_pose["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.cfg["ARCH"]["pose_model_input"] == "all" else 2)

            self.models_pose["pose"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())


        ### I do not implement predictive mask

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.cfg["train_params"]["lr"][0], weight_decay=self.cfg["train_params"]["weight_decay"])
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.cfg["train_params"]["lr"][3], weight_decay=self.cfg["train_params"]["weight_decay"])

        self.model_lr_scheduler = ChainedScheduler(
                            self.model_optimizer,
                            T_0=int(self.cfg["train_params"]["lr"][2]),
                            T_mul=1,
                            eta_min=self.cfg["train_params"]["lr"][1],
                            last_epoch=-1,
                            max_lr=self.cfg["train_params"]["lr"][0],
                            warmup_steps=0,
                            gamma=0.9
                        )
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.cfg["train_params"]["lr"][5]),
            T_mul=1,
            eta_min=self.cfg["train_params"]["lr"][4],
            last_epoch=-1,
            max_lr=self.cfg["train_params"]["lr"][3],
            warmup_steps=0,
            gamma=0.9
        )

        if self.cfg["train_params"]["load_weights_folder"] is not None:
            self.load_weights_folder = self.cfg["train_params"]["load_weights_folder"]
            self.load_model()

        # if self.cfg["train_params"]["mypretrain"] is not None:
        #     self.mypretrain = self.cfg["train_params"]["mypretrain"]
        #     self.load_pretrain()

        print("Training model named:\n  ", self.cfg["name"])
        print("Models and tensorboard events files are saved to:\n  ", self.cfg["log_dir"])
        print("Training is using:\n  ", self.device)

        # data
        self.split = self.cfg["dataset"]["split"]
        
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        
        self.dataset = datasets_dict[self.cfg["dataset"]["name"]]


        fpath = os.path.join(os.path.dirname(__file__), "splits", self.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.cfg["png"] else '.jpg'
        
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.cfg["train_params"]["batch_size"] * self.cfg["train_params"]["num_epochs"]

        ### Set Training dataset
        train_dataset = self.dataset(
             self.cfg["dataset"]["data_path"], train_filenames, self.cfg["dataset"]["height"], self.cfg["dataset"]["width"],
            self.frame_ids, 4, is_train=True, img_ext=img_ext)
        
        ### Set Val dataset
        self.train_loader = DataLoader(
            train_dataset, self.cfg["train_params"]["batch_size"], True,
            num_workers=self.cfg["train_params"]["num_workers"], pin_memory=True, drop_last=True)
        
        
        val_dataset = self.dataset(
            self.cfg["dataset"]["data_path"], val_filenames,  self.cfg["dataset"]["height"], self.cfg["dataset"]["width"],
            self.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.cfg["train_params"]["batch_size"], True,
            num_workers=self.cfg["train_params"]["num_workers"], pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)


        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.cfg["train_params"]["no_ssim"]:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        
        
        self.backproject_depth = {}
        self.project_3d = {}
        
        for scale in self.scales:
            h = self.cfg["dataset"]["height"] // (2 ** scale)
            w =  self.cfg["dataset"]["width"] // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.cfg["train_params"]["batch_size"], h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.cfg["train_params"]["batch_size"], h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))


    # def load_pretrain(self):
    #     self.mypretrain = os.path.expanduser(self.mypretrain)
    #     path = self.mypretrain
    #     model_dict = self.models["encoder"].state_dict()
    #     pretrained_dict = torch.load(path)['model']
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
    #     model_dict.update(pretrained_dict)
    #     self.models["encoder"].load_state_dict(model_dict)
    #     print('mypretrain loaded.')

    def load_model(self):
        """Load model(s) from disk
        """
        self.load_weights_folder = os.path.expanduser(self.load_weights_folder)

        assert os.path.isdir(self.load_weights_folder), \
            "Cannot find folder {}".format(self.load_weights_folder)
        print("loading model from folder {}".format(self.load_weights_folder))

        for n in self.cfg["train_params"]["models_to_load"]:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.load_weights_folder, "{}.pth".format(n))

            if n in ['pose_encoder', 'pose']:
                model_dict = self.models_pose[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models_pose[n].load_state_dict(model_dict)
            else:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        # loading adam state

        optimizer_load_path = os.path.join(self.load_weights_folder, "adam.pth")
        optimizer_pose_load_path = os.path.join(self.load_weights_folder, "adam_pose.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            optimizer_pose_dict = torch.load(optimizer_pose_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
            self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()



    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            
            # Do not implement v1 multi-scale
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
            source_scale = 0
            
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            
            
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.cfg["disable_automasking"]:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]


    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.cfg["train_params"]["num_epochs"]):
            self.run_epoch()
            if (self.epoch + 1) % self.cfg["train_params"]["save_frequency"] == 0:
                self.save_model()


    

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        self.model_lr_scheduler.step()
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.cfg["train_params"]["log_frequency"] == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
    
    
    def val(self):
        """Validate the model on a single minibatch
        """
        print("Val")
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # Only sperate mode for pose net
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

            for f_i in self.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.frame_ids if i != "s"], 1)

                if self.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            
            axisangle, translation = self.models_pose["pose"](pose_inputs)

            for i, f_i in enumerate(self.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        
        outputs = {}
        features = None # No shared mode
        depth_output = self.models["depth"](inputs["color_aug", 0, 0]) # [1/4 depth map, 1/2 depth map, full res depth map]
        
        for i in range(2, -1, -1):
            if i in self.scales:
                outputs[("disp", i)] = depth_output[2 - i]

        # if self.opt.predictive_mask:
        #     outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses
    


    def compute_losses(self, inputs, outputs, use_mean=False):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            # if self.opt.v1_multiscale:
            #     source_scale = scale
            # else:
            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.cfg["disable_automasking"]:
                identity_reprojection_losses = []
                for frame_id in self.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]    #?
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if use_mean:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            # elif self.opt.predictive_mask:
            #     # use the predicted mask
            #     mask = outputs["predictive_mask"]["disp", scale]
            #     if not self.opt.v1_multiscale:
            #         mask = F.interpolate(
            #             mask, [self.opt.height, self.opt.width],
            #             mode="bilinear", align_corners=False)

            #     reprojection_losses *= mask

            #     # add a loss pushing mask to 1 (using nn.BCELoss for stability)
            #     weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
            #     loss += weighting_loss.mean()

            if use_mean:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.cfg["disable_automasking"]:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.cfg["disable_automasking"]:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.cfg["train_params"]["disparity_smoothness"] * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred) # Change loss? L1, L2
        l1_loss = abs_diff.mean(1, True)

        if self.cfg["train_params"]["no_ssim"]:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.cfg["train_params"]["batch_size"] / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.cfg["train_params"]["batch_size"])):  # write a maxmimum of four images
            for s in self.scales:
                for frame_id in self.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                # if self.opt.predictive_mask:
                #     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                #         writer.add_image(
                #             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                #             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                #             self.step)

                if not self.cfg["disable_automasking"]:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)


    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.height
                to_save['width'] = self.width
                to_save['use_stereo'] = self.cfg["train_params"]["use_stereo"]
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)