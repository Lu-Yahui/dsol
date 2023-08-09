#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <yaml-cpp/yaml.h>

#include <iomanip>
#include <string>

#include "sv/dsol/extra.h"
#include "sv/dsol/odom.h"
#include "sv/util/dataset.h"
#include "sv/util/logging.h"
#include "sv/util/ocv.h"

ABSL_FLAG(std::string,
          config_dir,
          "./config/dsol_kitti.yaml",
          "DSOL config file for Kitti dataset");

namespace sv::dsol {

struct PointXYZIRGB {
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};
  float i{0.0F};
  uint8_t r{0U};
  uint8_t g{0U};
  uint8_t b{0U};
};

class DsolKitti {
 public:
  DsolKitti(const std::string& config_dir) : config_dir_(config_dir) {}

  void Init() {
    InitDataSet();
    InitMotion();
    InitOdom();
  }

  void Run() {
    for (int i = 0; i < dataset_.size(); ++i) {
      auto timestamp = dataset_.Get(DataType::kTime, i, 0).at<double>(0, 0);
      auto left_image = dataset_.Get(DataType::kImage, i, 0);
      auto right_image = dataset_.Get(DataType::kImage, i, 1);

      LOG(INFO) << "Num: " << i << ", Timestamp: " << timestamp;

      if (!odom_->camera.Ok()) {
        const auto& intrin = dataset_.Get(DataType::kIntrin, i);
        const auto& camera =
            Camera::FromMat({left_image.cols, left_image.rows}, intrin);
        odom_->SetCamera(camera);
      }

      double dt{0.0};
      if (last_timestamp_ > 0.0) {
        dt = timestamp - last_timestamp_;
      }

      Sophus::SE3d dtf_pred;
      if (dt > 0.0) {
        dtf_pred = motion_->PredictDelta(dt);
      }

      const auto& status = odom_->Estimate(left_image, right_image, dtf_pred);
      LOG(INFO) << status.Repr();

      if (status.track.ok) {
        motion_->Correct(status.Twc(), dt);
      }

      if (status.map.remove_kf) {
        LOG(INFO) << "Saving KF...";
        std::ofstream ofs(out_dir_ + "/" + std::to_string(i) + ".csv");
        const auto& kf = odom_->window.MargKf();
        const auto& points = KeyFrameToPoints(kf, 75.0);
        for (const auto& p : points) {
          ofs << std::setprecision(16) << p.x << " " << p.y << " " << p.z << " "
              << p.i << " " << static_cast<int>(p.r) << " "
              << static_cast<int>(p.g) << " " << static_cast<int>(p.b)
              << std::endl;
        }
      }

      last_timestamp_ = timestamp;
    }
  }

 private:
  void InitDataSet() {
    const auto& yaml = YAML::LoadFile(config_dir_);

    const auto& dataset_yaml = yaml["dataset"];
    out_dir_ = dataset_yaml["out_dir"].as<std::string>();

    std::string data_dir = dataset_yaml["data_dir"].as<std::string>();
    const std::string seq = dataset_yaml["seq"].as<std::string>();
    data_dir = data_dir + "/" + seq;

    dataset_ = KittiOdom(data_dir,
                         dataset_yaml["left_image"].as<std::string>(),
                         dataset_yaml["right_image"].as<std::string>(),
                         dataset_yaml["calib_file"].as<std::string>(),
                         dataset_yaml["pose_dir"].as<std::string>(),
                         dataset_yaml["time_dir"].as<std::string>());
    LOG(INFO) << dataset_.Repr();
  }

  std::vector<PointXYZIRGB> KeyFrameToPoints(const Keyframe& keyframe,
                                             double max_depth) {
    const auto& points = keyframe.points();
    const auto& patches = keyframe.patches().front();
    auto color_l = keyframe.color_l().clone();

    std::vector<PointXYZIRGB> points_out;
    for (int gr = 0; gr < points.rows(); ++gr) {
      for (int gc = 0; gc < points.cols(); ++gc) {
        const auto& point = points.at(gr, gc);
        if (!point.InfoMax() || (1.0 / point.idepth()) > max_depth) {
          continue;
        }
        CHECK(point.PixelOk());
        CHECK(point.DepthOk());

        // transform to fixed frame
        const Eigen::Vector3f p_w = (keyframe.Twc() * point.pt()).cast<float>();
        const auto& patch = patches.at(gr, gc);
        const auto& color = keyframe.color_l().at<cv::Vec3b>(point.px());

        PointXYZIRGB p_out{};
        p_out.x = p_w.x();
        p_out.y = p_w.y();
        p_out.z = p_w.z();
        p_out.i = static_cast<float>(patch.vals[0] / 255.0);
        p_out.r = static_cast<uint8_t>(color(2));
        p_out.g = static_cast<uint8_t>(color(1));
        p_out.b = static_cast<uint8_t>(color(0));

        cv::circle(color_l, point.px(), 3, cv::Scalar(0, 0, 255, 255), 1);

        points_out.push_back(p_out);
      }
    }

    // cv::imwrite("/tmp/test.png", color_l);
    cv::imshow("color", color_l);
    cv::waitKey(1);

    return points_out;
  }

  void InitOdom() {
    const auto& yaml = YAML::LoadFile(config_dir_);

    // Init odom
    OdomCfg odom_cfg{};
    const auto& odom_yaml = yaml["odom"];
    odom_cfg.vis = odom_yaml["vis"].as<int>();
    odom_cfg.marg = odom_yaml["marg"].as<bool>();
    odom_cfg.num_kfs = odom_yaml["num_kfs"].as<int>();
    odom_cfg.num_levels = odom_yaml["num_levels"].as<int>();
    odom_cfg.vis_min_depth = odom_yaml["vis_min_depth"].as<double>();
    odom_cfg.reinit = odom_yaml["reinit"].as<bool>();
    odom_cfg.init_depth = odom_yaml["init_depth"].as<bool>();
    odom_cfg.init_stereo = odom_yaml["init_stereo"].as<bool>();
    odom_cfg.init_align = odom_yaml["init_align"].as<bool>();
    odom_ = std::make_unique<DirectOdometry>(odom_cfg);

    // init pixel selector
    SelectCfg select_cfg{};
    const auto& select_yaml = yaml["select"];
    select_cfg.sel_level = select_yaml["sel_level"].as<int>();
    select_cfg.cell_size = select_yaml["cell_size"].as<int>();
    select_cfg.min_grad = select_yaml["min_grad"].as<int>();
    select_cfg.max_grad = select_yaml["max_grad"].as<int>();
    select_cfg.nms_size = select_yaml["nms_size"].as<int>();
    select_cfg.min_ratio = select_yaml["min_ratio"].as<double>();
    select_cfg.max_ratio = select_yaml["max_ratio"].as<double>();
    select_cfg.reselect = select_yaml["reselect"].as<bool>();
    odom_->selector = PixelSelector(select_cfg);

    // init stereo matcher
    StereoCfg stereo_matcher_cfg{};
    const auto& stereo_yaml = yaml["stereo"];
    stereo_matcher_cfg.half_rows = stereo_yaml["half_rows"].as<int>();
    stereo_matcher_cfg.half_cols = stereo_yaml["half_cols"].as<int>();
    stereo_matcher_cfg.match_level = stereo_yaml["match_level"].as<int>();
    stereo_matcher_cfg.refine_size = stereo_yaml["refine_size"].as<int>();
    stereo_matcher_cfg.min_zncc = stereo_yaml["min_zncc"].as<double>();
    stereo_matcher_cfg.min_depth = stereo_yaml["min_depth"].as<double>();
    odom_->matcher = StereoMatcher(stereo_matcher_cfg);

    // init frame align
    DirectCfg align_cfg{};
    const auto& align_yaml = yaml["align"];
    align_cfg.optm.init_level = align_yaml["init_level"].as<int>();
    align_cfg.optm.max_iters = align_yaml["max_iters"].as<int>();
    align_cfg.optm.max_xs = align_yaml["max_xs"].as<double>();
    align_cfg.cost.affine = align_yaml["affine"].as<bool>();
    align_cfg.cost.stereo = align_yaml["stereo"].as<bool>();
    align_cfg.cost.c2 = align_yaml["c2"].as<int>();
    align_cfg.cost.dof = align_yaml["dof"].as<int>();
    align_cfg.cost.max_outliers = align_yaml["max_outliers"].as<int>();
    align_cfg.cost.grad_factor = align_yaml["grad_factor"].as<double>();
    align_cfg.cost.min_depth = align_yaml["min_depth"].as<double>();
    odom_->aligner = FrameAligner(align_cfg);

    // init bundle adjuster
    DirectCfg ba_cfg{};
    const auto& ba_yaml = yaml["adjust"];
    ba_cfg.optm.init_level = ba_yaml["init_level"].as<int>();
    ba_cfg.optm.max_iters = ba_yaml["max_iters"].as<int>();
    ba_cfg.optm.max_xs = ba_yaml["max_xs"].as<double>();
    ba_cfg.cost.affine = ba_yaml["affine"].as<bool>();
    ba_cfg.cost.stereo = ba_yaml["stereo"].as<bool>();
    ba_cfg.cost.c2 = ba_yaml["c2"].as<int>();
    ba_cfg.cost.dof = ba_yaml["dof"].as<int>();
    ba_cfg.cost.max_outliers = ba_yaml["max_outliers"].as<int>();
    ba_cfg.cost.grad_factor = ba_yaml["grad_factor"].as<double>();
    ba_cfg.cost.min_depth = ba_yaml["min_depth"].as<double>();
    odom_->adjuster = BundleAdjuster(ba_cfg);

    // init color map
    odom_->cmap = GetColorMap("jet");

    LOG(INFO) << odom_->Repr();
  }

  void InitMotion() { motion_ = std::make_unique<MotionModel>(0.5); }

 private:
  // DSOL config file
  std::string config_dir_;
  // dataset
  std::string out_dir_;

  Dataset dataset_;
  std::unique_ptr<MotionModel> motion_;
  std::unique_ptr<DirectOdometry> odom_;

  double last_timestamp_{0.0};
};

}  // namespace sv::dsol

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string config_dir = absl::GetFlag(FLAGS_config_dir);
  sv::dsol::DsolKitti dsol(config_dir);
  dsol.Init();
  dsol.Run();
  return 0;
}
