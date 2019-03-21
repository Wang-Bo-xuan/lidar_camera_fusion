#include <cmath>
#include <ros/ros.h>
#include "sensor_msgs/Image.h"
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>
using namespace cv;

ros::Publisher points_pub;
ros::Publisher rgb_image_pub;
cv::Mat rect_view;

std::string image_topic;
std::string lidar_topic;
std::string calibration_filename;
std::string image_out_topic;
std::string lidar_out_topic;
std::string publish_frame;

bool get_image;
bool get_lidar;

double cameraExtrinsicCoef[16] =  { -6.2309509256386342e-02, -2.4790966187969110e-02,
9.9774893287424526e-01, -1.8460420409124567e-01,
-9.9805683184592064e-01, 1.2549028088513214e-03,
-6.2297557132074355e-02, 7.0578993330759751e-03,
2.9233869406214419e-04, -9.9969186913488861e-01,
-2.4820985550147778e-02, -1.3158911113547375e-01, 0., 0., 0., 1. };

double cameraCoef[9] =   { 4.6759942166183509e+02, 0., 3.1303526693575287e+02, 0.,
4.6340598854771986e+02, 2.5928859730584895e+02, 0., 0., 1. };

double distCoef[5] = { -8.4648413157903932e-02, 5.3506369299288803e-01,
1.1250841849871174e-02, -1.7186883524251120e-03,
-9.1759236731969551e-01 };

cv::Mat cameraExtrinsicMat = cv::Mat(3, 4, CV_64F, cameraExtrinsicCoef);
cv::Mat cameraMat = cv::Mat(3, 3, CV_64F, cameraCoef);
cv::Mat distMat = cv::Mat(5, 1, CV_64F, distCoef);
cv::Mat rotateMat = cameraExtrinsicMat(cv::Rect(0, 0, 3, 3));
cv::Mat invT = -rotateMat.t() * (cameraExtrinsicMat(cv::Rect(3, 0, 1, 3)));
cv::Mat translateMat = invT.t();
pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudIn(new pcl::PointCloud<pcl::PointXYZ>());

void cameraCallback(const sensor_msgs::Image::ConstPtr& img)
{
  get_image = true;

  cv_bridge::CvImagePtr cv_cam = cv_bridge::toCvCopy(img, "8UC3");
  cvtColor(cv_cam->image, rect_view, CV_BGR2RGB);
  cvtColor(rect_view,rect_view,CV_BGR2RGB);
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", rect_view).toImageMsg();

  rgb_image_pub.publish(msg);
}

void lidarCallback(const sensor_msgs::PointCloud2::ConstPtr& scan)
{
  get_lidar = true;

  pcl::fromROSMsg(*scan, *laserCloudIn);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colour_laser_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

  for(unsigned int i = 0; i < laserCloudIn->size(); i++)
  {
    float x3d = laserCloudIn->at(i).x;
    float y3d = laserCloudIn->at(i).y;
    float z3d = laserCloudIn->at(i).z;

    pcl::PointXYZRGB p;
    p.x = laserCloudIn->at(i).x;
    p.y = laserCloudIn->at(i).y;
    p.z = laserCloudIn->at(i).z;
    p.r = 0;
    p.g = 0;
    p.b = 0;

    cv::Mat point(1, 3, CV_64F);
    cv::Point2d imagepoint;
    double fp[3] = {p.x, p.y, p.z};
    for (int i = 0; i < 3; i++)
    {
      point.at<double>(i) = translateMat.at<double>(i);
      for (int j = 0; j < 3; j++)
      {
        point.at<double>(i) += double(fp[j]) * rotateMat.at<double>(j, i);
      }
    }

    if (point.at<double>(2) > 1)
    {
      double tmpx = point.at<double>(0) / point.at<double>(2);
      double tmpy = point.at<double>(1) / point.at<double>(2);
      double r2 = tmpx * tmpx + tmpy * tmpy;
      double tmpdist = 1 + distMat.at<double>(0) * r2 + distMat.at<double>(1) * r2 * r2 + distMat.at<double>(4) * r2 * r2 * r2;
      imagepoint.x = tmpx * tmpdist + 2 * distMat.at<double>(2) * tmpx * tmpy + distMat.at<double>(3) * (r2 + 2 * tmpx * tmpx);
      imagepoint.y = tmpy * tmpdist + distMat.at<double>(2) * (r2 + 2 * tmpy * tmpy) + 2 * distMat.at<double>(3) * tmpx * tmpy;
      imagepoint.x = cameraMat.at<double>(0, 0) * imagepoint.x + cameraMat.at<double>(0, 2);
      imagepoint.y = cameraMat.at<double>(1, 1) * imagepoint.y + cameraMat.at<double>(1, 2);
      int px = int(imagepoint.x + 0.5);
      int py = int(imagepoint.y + 0.5);

      if (0 <= px && px < rect_view.size().width && 0 <= py && py < rect_view.size().height)
      {
        p.r = rect_view.at<cv::Vec3b>(py, px)[0];
        p.g = rect_view.at<cv::Vec3b>(py, px)[1];
        p.b = rect_view.at<cv::Vec3b>(py, px)[2];
      }
    }

//    if(!(p.r == 0 && p.g == 0 && p.b == 0))
    {
      colour_laser_cloud->push_back(p);
    }
  }

  sensor_msgs::PointCloud2 scan_color = sensor_msgs::PointCloud2();
  pcl::toROSMsg(*colour_laser_cloud, scan_color);
  scan_color.header.frame_id = publish_frame.data();
  points_pub.publish(scan_color);
}

int main(int argc,char *argv[])
{
  ros::init(argc, argv, "fusion_node");
  ros::NodeHandle n;
  ros::NodeHandle pnh("~");
  ros::Rate loop_rate(20.0);

  pnh.param<std::string>("image_topic",image_topic,"/image_raw");
  pnh.param<std::string>("lidar_topic",lidar_topic,"/velodyne_points");
  pnh.param<std::string>("calibration_filename",calibration_filename,"");

  pnh.param<std::string>("lidar_out_topic",lidar_out_topic,"/color_points");
  pnh.param<std::string>("image_out_topic",image_out_topic,"/rect_image");
  pnh.param<std::string>("publish_frame",publish_frame,"velodyne");

  if(calibration_filename.empty())
  {
    ROS_WARN("could not find the calibration file ...");
  }
//  else
//  {
//    YAML::Node yamlConfig = YAML::LoadFile(calibration_filename);
//    int int_param = yamlConfig["int_param"].as<int>();
//    std::cout << "  node size: " << yamlConfig.size() << std::endl;
//    std::cout << yamlConfig["bool_param"].as<bool>() << "\n";
//    yamlConfig["bool_param"] = !yamlConfig["bool_param"].as<bool>();
//    yamlConfig["str_param"] = "test";
//    std::ofstream file;
//    file.open(fin);
//    file.flush();
//    file << yamlConfig;
//  }

  get_lidar = false;
  get_image = false;

  ros::Subscriber image_sub = n.subscribe(image_topic.data(), 1, cameraCallback);
  ros::Subscriber lidar_sub = n.subscribe(lidar_topic.data(), 1, lidarCallback);

  points_pub = n.advertise<sensor_msgs::PointCloud2>(lidar_out_topic.data(), 10);
  rgb_image_pub = n.advertise<sensor_msgs::Image>(image_out_topic.data(), 20);

  while(ros::ok())
  {
    if(!get_lidar)
    {
      ROS_WARN("no lidar input");
    }

    if(!get_image)
    {
      ROS_WARN("no image input");
    }

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
