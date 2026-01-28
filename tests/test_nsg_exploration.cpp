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
    std::cout << "open file error" << filename << std::endl;
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

template<typename... Args>
std::string string_format(const char* fmt, Args... args)
{
    size_t size = snprintf(nullptr, 0, fmt, args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt, args...);
    return buf;
}

int main(int argc, char** argv) {

  #if defined(__AVX__)
    std::cout << "use AVX2  ..." << std::endl;
  #elif defined(__SSE2__)
    std::cout << "use SSE  ..." << std::endl;
  #else
    std::cout << "use arch  ..." << std::endl;
  #endif
  std::cout << "DATA_ALIGN_FACTOR " << DATA_ALIGN_FACTOR << std::endl;

  #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
  #endif

  uint32_t k = 1000;

  // --------------------------------------------------- SIFT1M -------------------------------------------------
  // auto nsg_file         = R"(e:/Data/Feature/SIFT1M/nsg/sift_L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";
  // auto object_file      = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  // auto query_file       = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_explore_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_explore_ground_truth.ivecs)";
  // auto entry_node_file  = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_explore_entry_vertex.ivecs)";

  // --------------------------------------------------- Glove -------------------------------------------------
  // auto nsg_file         = R"(e:/Data/Feature/Glove/nsg/glove-100_L50_R70_C500_EfaK400_L420_It12_S15_R200.nsg)";
  // auto object_file      = R"(e:/Data/Feature/Glove/glove-100/glove-100_base.fvecs)";
  // auto query_file       = R"(e:/Data/Feature/Glove/glove-100/glove-100_explore_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/Glove/glove-100/glove-100_explore_ground_truth.ivecs)";
  // auto entry_node_file  = R"(e:/Data/Feature/Glove/glove-100/glove-100_explore_entry_vertex.ivecs)";

  // --------------------------------------------------- Enron -------------------------------------------------
  // auto nsg_file         = R"(e:/Data/Feature/Enron/nsg/L150_R60_C600_efaK200_L200_It7_S25_R200.nsg)";
  // auto object_file      = R"(e:/Data/Feature/Enron/enron/enron_base.fvecs)";
  // auto query_file       = R"(e:/Data/Feature/Enron/enron/enron_explore_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/Enron/enron/enron_explore_ground_truth.ivecs)";
  // auto entry_node_file  = R"(e:/Data/Feature/Enron/enron/enron_explore_entry_vertex.ivecs)";

  // // --------------------------------------------------- Audio -------------------------------------------------
  // auto nsg_file         = R"(e:/Data/Feature/Audio/nsg/L200_R30_C600_efaK200_L230_It5_S10_R100.nsg)";
  // auto object_file      = R"(e:/Data/Feature/Audio/audio/audio_base.fvecs)";
  // auto query_file       = R"(e:/Data/Feature/Audio/audio/audio_explore_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/Audio/audio/audio_explore_ground_truth.ivecs)";
  // auto entry_node_file  = R"(e:/Data/Feature/Audio/audio/audio_explore_entry_vertex.ivecs)";

  // --------------------------------------------------- Deep1M -------------------------------------------------
  auto nsg_file         = R"(e:/Data/Feature/Deep1M/nsg/L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";
  auto object_file      = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_base.fvecs)";
  auto query_file       = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_explore_query.fvecs)";
  auto groundtruth_file = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_explore_ground_truth.ivecs)";
  auto entry_node_file  = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_explore_entry_vertex.ivecs)";


  // load feature vectors
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(object_file, data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim); // align the data before build
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading base data" << std::endl;

  // create the index
  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
  index.Load(nsg_file);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading base data" << std::endl;

  // query data
  float* query_data = NULL;
  unsigned query_num, query_dim;
  load_data(query_file, query_data, query_num, query_dim);

  // query ground truth
  float* groundtruth_f = NULL;
  unsigned groundtruth_num, groundtruth_dim;
  load_data(groundtruth_file, groundtruth_f, groundtruth_num, groundtruth_dim);
  const auto ground_truth = (uint32_t*)groundtruth_f; // not very clean, works as long as sizeof(int) == sizeof(float)
  const auto answers = get_ground_truth(ground_truth, groundtruth_num, groundtruth_dim, k);

  // load entry node
  float* entry_node_f = NULL;
  unsigned entry_node_num, entry_node_dim;
  load_data(entry_node_file, entry_node_f, entry_node_num, entry_node_dim);
  const auto entry_node = (uint32_t*)entry_node_f; // not very clean, works as long as sizeof(int) == sizeof(float)

  // try differen P_search parameters
  uint32_t k_factor = k/10;
  for (uint32_t f = 0; f <= 3; f++, k_factor *= 10) {
    for (uint32_t i = (f == 0) ? 1 : 2; i < 11; i++) {         
      const auto max_distance_count = ((f == 0) ? (k + k_factor * (i-1)) : (k_factor * i));

      auto tmp = std::vector<unsigned>(k);
      auto time_begin = std::chrono::steady_clock::now();

      size_t correct = 0;
      for (unsigned i = 0; i < query_num; i++) {
        auto entry_node_index = entry_node[i * entry_node_dim];
        index.Explore(entry_node_index, data_load, k, tmp.data(), max_distance_count);

        // compare answer with ann
        auto answer = answers[i];
        for (size_t r = 0; r < k; r++) 
          if (answer.find(tmp[r]) != answer.end()) correct++;
      }

      auto recall = 1.0f * correct / (query_num * k);
      auto time_end = std::chrono::steady_clock::now();
      auto time_us_per_query = (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / query_num;
      std::cout << string_format("k and p %5d, max_distance_count %6d, recall %.4f, time_us_per_query %6d\n", k, max_distance_count, recall, time_us_per_query);
    }
  }

  return 0;
}