#include <ros/ros.h>
#include <std_msgs/Int8MultiArray.h>
#include <sensor_msgs/JointState.h>
#include <vector>
#include <mutex>
#include <thread>
#include <array>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <memory>
#include <chrono>
#include <Eigen/Dense>

typedef std::chrono::duration<double, std::ratio<1> > second;
typedef std::chrono::high_resolution_clock hd_clock;

class EnvDistCheck 
{
public:
    EnvDistCheck(ros::NodeHandle& nh) : N_(10), device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
    {
        // Subscribe to the masked_voxel and joint_states topics
        voxel_sub_ = nh.subscribe("/masked_voxel", 10, &EnvDistCheck::voxelCallback, this);
        joint_sub_ = nh.subscribe("/joint_states", 10, &EnvDistCheck::jointCallback, this);

        // Initialize voxel and joint data as empty
        voxel_data_.clear();
        joint_data_.fill(0);

        std::string model_path;
        nh.getParam("/env_dist_check_node/model_file", model_path);
        try 
        {
            model_ = torch::jit::load(model_path);
            model_.to(device_);
        } 
        catch (const c10::Error& e) 
        {
            ROS_WARN_STREAM("Error loading the model");
        }
        ROS_INFO("Model loaded successfully");

        x_q_ = torch::zeros({1, 7}, torch::kFloat32).to(device_);
        x_occ_ = torch::zeros({1, 1, 36, 36, 36}, torch::kFloat32).to(device_);
    }

    void voxelCallback(const std_msgs::Int8MultiArray::ConstPtr& data) 
    {
        // std::lock_guard<std::mutex> lock(data_mutex_);
        
        // Get the dimensions from the layout
        int dim_x = data->layout.dim[0].size;
        int dim_y = data->layout.dim[1].size;
        int dim_z = data->layout.dim[2].size;

        // Resize voxel_data_ to match the received dimensions
        voxel_data_.resize(dim_x * dim_y * dim_z);
        
        // Copy data to the voxel_data_ vector
        std::copy(data->data.begin(), data->data.end(), voxel_data_.begin());
        data_mutex_.lock();
        x_occ_ = torch::from_blob(voxel_data_.data(), {1, 1, 36, 36, 36}, torch::kFloat32).clone().to(device_);
        data_mutex_.unlock();
    }

    void jointCallback(const sensor_msgs::JointState::ConstPtr& data) 
    {
        // std::lock_guard<std::mutex> lock(data_mutex_);

        // Define the joint names to track
        std::array<std::string, 7> conf_joint_names = 
        {
            "panda_joint1", "panda_joint2", "panda_joint3", 
            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
        };

        // Extract the positions of the specified joints
        for (size_t i = 0; i < data->name.size(); ++i) {
            for (size_t j = 0; j < conf_joint_names.size(); ++j) 
            {
                if (data->name[i] == conf_joint_names[j]) 
                {
                    joint_data_[j] = static_cast<float>(data->position[i]);
                    break;
                }
            }
        }
        data_mutex_.lock();
        x_q_ = torch::from_blob(joint_data_.data(), {1, 7}, torch::kFloat32).clone().to(device_);
        data_mutex_.unlock();
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> modelForward() 
    {
        // std::lock_guard<std::mutex> lock(data_mutex_);

        if (joint_data_.empty() || voxel_data_.empty()) 
        {
            return std::make_pair(Eigen::VectorXd(), Eigen::MatrixXd());
        }

        try 
        {
            auto start = hd_clock::now();

            // // Convert joint_data_ and voxel_data_ to torch tensors
            // auto tensor_start = hd_clock::now();
            // at::Tensor x_q = torch::empty({1, 7}, torch::kFloat32).to(device_);
            // at::Tensor x_occ = torch::empty({1, 1, 36, 36, 36}, torch::kFloat32).to(device_);
            // if(data_mutex_.try_lock())
            // {
            //     x_occ = torch::from_blob(voxel_data_.data(), {1, 1, 36, 36, 36}, torch::kFloat32).clone().to(device_);
            //     x_q = torch::from_blob(joint_data_.data(), {1, 7}, torch::kFloat32).clone().to(device_);
            //     data_mutex_.unlock();
            // }
            // auto tensor_end = hd_clock::now();
            // ROS_INFO_STREAM("Tensor Conversion[Hz]: " << 1./(std::chrono::duration_cast<second>(tensor_end - tensor_start).count()));


            // Repeat tensors to match batch size N_
            auto repeat_start = hd_clock::now();
            x_q_ = x_q_.repeat({N_, 1});
            x_occ_ = x_occ_.repeat({N_, 1, 1, 1, 1});
            auto repeat_end = hd_clock::now();
            // ROS_INFO_STREAM("Tensor Repeat[Hz]    : " << 1./(std::chrono::duration_cast<second>(repeat_end - repeat_start).count()));

            // Set requires_grad_ to true for x_q to enable gradient computation
            x_q_.set_requires_grad(true);

            // Forward pass through the model
            auto forward_start = hd_clock::now();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(x_q_);
            inputs.push_back(x_occ_);

            // torch::Tensor output = model_.forward(inputs).toTensor().cpu();
            torch::Tensor output = model_.forward(inputs).toTensor();
            auto forward_end = hd_clock::now();
            ROS_INFO_STREAM("Model Forward Pass[Hz]: " << 1./(std::chrono::duration_cast<second>(forward_end - forward_start).count()));


            // Calculate the Jacobian
            auto jacobian_start = hd_clock::now();
            // int output_size = output.size(0);
            // Eigen::MatrixXf jacobian_pred_T(x_q_.size(1), output_size);
            // for (int i = 0; i < output_size; ++i) 
            // {
            //     at::Tensor scalar_output = output[i][0];  // output[i] is a scalar (since output is [10, 1])
    
            //     // Perform backward pass for each scalar output to compute the gradient
            //     scalar_output.backward(torch::Tensor(), true);

            //     // Ensure the gradient is contiguous
            //     at::Tensor grad = x_q_.grad()[i].cpu();

            //     // Use memcpy to copy the gradient into the Eigen matrix (1D memory block)
            //     std::memcpy(jacobian_pred_T.col(i).data(), grad.data_ptr<float>(), 7 * sizeof(float));

            //     // Reset gradients for the next iteration
            //     x_q_.grad().zero_();
            // }

            // GPU에서 병렬로 Jacobian 계산
            // grad_outputs를 std::vector로 래핑
            std::vector<torch::Tensor> grad_outputs = {torch::ones_like(output).to(device_)};

            // autograd::grad로 Jacobian 계산
            std::vector<torch::Tensor> jacobians = torch::autograd::grad(
                {output},                // Output tensor
                {x_q_},                  // Input tensor we want to differentiate
                grad_outputs,            // Gradient w.r.t the outputs (as a vector)
                true,                    // Create graph
                true                     // Retain graph
            );


            // Tensor를 Eigen 형태로 변환
            Eigen::MatrixXf jacobian_pred_T(7, output.size(0));
            std::memcpy(jacobian_pred_T.data(), jacobians[0].cpu().data_ptr<float>(), 7 * output.size(0) * sizeof(float));

            auto jacobian_end = hd_clock::now();
            // ROS_INFO_STREAM("Jacobian Calculation[Hz]: " <<1./ (std::chrono::duration_cast<second>(jacobian_end - jacobian_start).count()));

            // Convert the output tensor to a vector for easy handling
            auto memcpy_start = hd_clock::now();
            Eigen::VectorXf env_min_dist_pred(output.numel());
            std::memcpy(env_min_dist_pred.data(), output.cpu().data_ptr<float>(), output.numel() * sizeof(float));
            // auto memcpy_end = hd_clock::now();
            // ROS_INFO_STREAM("Memcpy[Hz]           : " << 1./(std::chrono::duration_cast<second>(memcpy_end - memcpy_start).count()));


            auto end = hd_clock::now();
            ROS_INFO_STREAM("Total ModelForward[Hz]: " <<1./ (std::chrono::duration_cast<second>(end - start).count()));

            // return env_min_dist_pred;
            return std::make_pair(env_min_dist_pred.cast<double>(), jacobian_pred_T.transpose().cast<double>());
        } 
        catch (const std::exception& e) 
        {
            ROS_ERROR_STREAM("Exception in modelForward: " << e.what());
            return std::make_pair(Eigen::VectorXd(), Eigen::MatrixXd());
        }
    }



private:
    ros::Subscriber voxel_sub_;
    ros::Subscriber joint_sub_;
    std::mutex data_mutex_;
    std::vector<float> voxel_data_;
    std::array<float, 7> joint_data_;
    at::Tensor x_q_;
    at::Tensor x_occ_;

    torch::jit::script::Module model_;
    int N_;
    torch::Device device_;
};

int main(int argc, char **argv) {
    // Initialize the ROS system
    ros::init(argc, argv, "env_dist_check_node");

    // Initialize the ROS node
    ros::NodeHandle nh;

    // Create an instance of the EnvDistCheck class
    EnvDistCheck env_dist_check(nh);
    
    // Create a loop rate object for 10Hz
    ros::Rate rate(10);

    while (ros::ok()) 
    {
        auto result = env_dist_check.modelForward();
        // if (!result.first.empty()) 
        if (result.first.size() > 0) 
        {
            ROS_INFO_STREAM("env_pred size: " << result.first.size());
            ROS_INFO_STREAM("env_pred: " << result.first(0) << " cm");
            // Uncomment below to print the Jacobian (it could be large)
            ROS_INFO_STREAM("Jacobian size: " << result.second.rows() << ", " << result.second.cols());
            // ROS_INFO_STREAM("Jacobian: " << result.second.row(0));
            // std::cout << result.second << std::endl;
            std::cout << "==============================================" << std::endl;
        }
        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}
