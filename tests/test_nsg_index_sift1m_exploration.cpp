//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

#include <vector>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

static void load_data(const char* filename, float*& data, unsigned& num, unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

static std::vector<std::unordered_set<uint32_t>> get_ground_truth(const uint32_t* ground_truth, const size_t ground_truth_size, const uint32_t ground_truth_dims, const size_t k)
{
    auto answers = std::vector<std::unordered_set<uint32_t>>(ground_truth_size);
    answers.reserve(ground_truth_size);
    for (int i = 0; i < ground_truth_size; i++)
    {
        auto& gt = answers[i];
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) 
            gt.insert(ground_truth[ground_truth_dims * i + j]);
    }

    return answers;
}

int main(int argc, char** argv) {

  std::cout << "DATA_ALIGN_FACTOR " << DATA_ALIGN_FACTOR << std::endl;
  #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
  #endif

  #ifdef __AVX__
    std::cout << "__AVX__ is set" << std::endl;
  #endif

  auto object_file      = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  auto nsg_file         = R"(c:/Data/Feature/SIFT1M/nsg/sift.nsg)";
  auto query_file       = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_explore_query.fvecs)";
  auto groundtruth_file = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_explore_ground_truth.ivecs)";
  auto entry_node_file  = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_explore_entry_node.ivecs)";
  uint32_t max_k = 1000;

  // load feature vectors
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(object_file, data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim); // align the data before build

  // create the index
  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
  index.Load(nsg_file);

  // query data
  float* query_data = NULL;
  unsigned query_num, query_dim;
  load_data(query_file, query_data, query_num, query_dim);

  // query ground truth
  float* groundtruth_f = NULL;
  unsigned groundtruth_num, groundtruth_dim;
  load_data(groundtruth_file, groundtruth_f, groundtruth_num, groundtruth_dim);
  const auto ground_truth = (uint32_t*)groundtruth_f; // not very clean, works as long as sizeof(int) == sizeof(float)

  // load entry node
  float* entry_node_f = NULL;
  unsigned entry_node_num, entry_node_dim;
  load_data(entry_node_file, entry_node_f, entry_node_num, entry_node_dim);
  const auto entry_node = (uint32_t*)entry_node_f; // not very clean, works as long as sizeof(int) == sizeof(float)

  // try differen P_search parameters
  /*auto steps = 100;
  for (size_t i = 1; i <= steps; i++) {
    const auto K = max_k/steps*i;
    const auto max_distance_computation_count = K;*/
  auto steps = 30;
  for (size_t i = 0; i <= steps; i++) {
    const auto K = max_k;
    const auto max_distance_count = K + (K/10 * i);

    const auto answers = get_ground_truth(ground_truth, groundtruth_num, groundtruth_dim, K);

    auto tmp = std::vector<unsigned>(K);
    auto time_begin = std::chrono::steady_clock::now();

    size_t correct = 0;
    for (unsigned i = 0; i < query_num; i++) {
      auto entry_node_index = entry_node[i * entry_node_dim];
      index.Explore(entry_node_index, data_load, K, tmp.data(), max_distance_count);

      // compare answer with ann
      auto answer = answers[i];
      for (size_t r = 0; r < K; r++) 
        if (answer.find(tmp[r]) != answer.end()) correct++;
    }

    auto recall = 1.0f * correct / (query_num * K);
    auto time_end = std::chrono::steady_clock::now();
    auto time_us_per_query = (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / query_num;
    std::cout << "k and p " << K << ", max_distance_count " << max_distance_count << ", recall " << recall << " time_us_per_query " << time_us_per_query << std::endl;
  }

  return 0;
}