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

  bool optimize_graph = false;  // uses normalized distances and DistanceFastL2 and prefetching
  unsigned K = 100;
  unsigned repeat_test = 1;

  // // ------------------------------------ SIFT1M -----------------------------------------
  // auto object_file      = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  // auto nsg_file         = R"(e:/Data/Feature/SIFT1M/nsg/sift_L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";
  // auto query_file       = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_groundtruth.ivecs)";
  // std::vector<unsigned> L_search_parameter = { 100, 120, 140, 170, 200, 300 };


  // ------------------------------------ Enron -----------------------------------------
  // auto object_file      = R"(e:/Data/Feature/Enron/enron/enron_base.fvecs)";
  // auto query_file       = R"(e:/Data/Feature/Enron/enron/enron_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/Enron/enron/enron_groundtruth_top1000.ivecs)";
  // auto nsg_file         = R"(e:/Data/Feature/Enron/nsg/L150_R60_C600_efaK200_L200_It7_S25_R200.nsg)";
  // // std::vector<unsigned> L_search_parameter = { 30, 40, 50, 70, 100, 150, 300, 400, 800 }; 
  // std::vector<unsigned> L_search_parameter = { 100, 150, 300, 400, 800 }; 
  // K = 100;
  // repeat_test = 20;

  // ------------------------------------ Audio -----------------------------------------
  // auto object_file      = R"(e:/Data/Feature/Audio/audio/audio_base.fvecs)";
  // auto query_file       = R"(e:/Data/Feature/Audio/audio/audio_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/Audio/audio/audio_groundtruth_top1000.ivecs)";
  // auto nsg_file         = R"(e:/Data/Feature/Audio/nsg/L200_R30_C600_efaK200_L230_It5_S10_R100.nsg)";
  // // std::vector<unsigned> L_search_parameter = { 20, 23, 30, 40, 70, 300 }; 
  // std::vector<unsigned> L_search_parameter = { 100, 150, 200, 300, 600 }; 
  // K = 100;
  // repeat_test = 20;

  // ------------------------------------ ImageNet1k -----------------------------------------
  // auto object_file      = R"(e:/Data/Feature/ImageNet1k/ImageNet1k/clip_base.fvecs)";
  // auto query_file       = R"(e:/Data/Feature/ImageNet1k/ImageNet1k/clip_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/ImageNet1k/ImageNet1k/clip_groundtruth.ivecs)";
  // auto nsg_file         = R"(e:/Data/Feature/ImageNet1k/nsg/sift_L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";
  // std::vector<unsigned> L_search_parameter = { 100, 150, 200, 300, 600 }; 
  // K = 100;
  // repeat_test = 1;

  // // ------------------------------------ Deep10M -----------------------------------------
  // auto object_file      = R"(e:/Data/Feature/Deep10M/deep10m/deep10m_base.fvecs)";
  // auto query_file       = R"(e:/Data/Feature/Deep10M/deep10m/deep10m_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/Deep10M/deep10m/deep10m_groundtruth.ivecs)";
  // auto nsg_file         = R"(e:/Data/Feature/Deep10M/nsg/deep10m_L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";
  // std::vector<unsigned> L_search_parameter = { 100, 150, 200, 300, 600 }; 
  // K = 100;
  // repeat_test = 1;

  // ------------------------------------ Deep1M -----------------------------------------
  auto object_file      = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_base.fvecs)";
  auto query_file       = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_query.fvecs)";
  auto groundtruth_file = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_groundtruth.ivecs)";
  auto nsg_file         = R"(e:/Data/Feature/Deep1M/nsg/L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";
  std::vector<unsigned> L_search_parameter = { 100, 150, 200, 300, 600 }; 
  K = 100;
  repeat_test = 1;


  // ---------------------------------------------------------------------------
  // -------------------------------- SSG --------------------------------------
  // ---------------------------------------------------------------------------
  // load feature vectors
  std::cout << "Load basedata and align" << std::endl;
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(object_file, data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim); // align the data before build
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading base data" << std::endl;

  // load the index
  std::cout << "Load graph" << std::endl;
  auto index =  efanna2e::IndexNSG(dim, points_num, efanna2e::L2, nullptr);
  index.Load(nsg_file);
  if(optimize_graph) 
    index.OptimizeGraph(data_load);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading graph" << std::endl;

  // query data
  float* query_data = NULL;
  unsigned query_num, query_dim;
  load_data(query_file, query_data, query_num, query_dim);

  // query ground truth
  float* groundtruth_f = NULL;
  unsigned groundtruth_num, groundtruth_dim;
  load_data(groundtruth_file, groundtruth_f, groundtruth_num, groundtruth_dim);
  const auto ground_truth = (uint32_t*)groundtruth_f; // not very clean, works as long as sizeof(int) == sizeof(float)
  const auto answers = get_ground_truth(ground_truth, groundtruth_num, groundtruth_dim, K);

  std::cout << "Evaluate TOP " << K << " graph (optimized=" << optimize_graph << ")" << std::endl;
  for (unsigned L_search : L_search_parameter) {

    if (L_search < K) {
      std::cout << "search_L cannot be smaller than search_K!" << std::endl;
      exit(-1);
    }

    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L_search);
    paras.Set<unsigned>("P_search", L_search);

    auto time_begin = std::chrono::steady_clock::now();

    size_t correct = 0;
    for (size_t t = 0; t < repeat_test; t++) {
      for (unsigned i = 0; i < query_num; i++) {
        std::vector<unsigned> tmp(K);
        if(optimize_graph) 
          index.SearchWithOptGraph(query_data + i * query_dim, K, paras, tmp.data());
        else
          index.Search(query_data + i * query_dim, data_load, K, paras, tmp.data());

        // compare answer with ann
        auto answer = answers[i];
        for (size_t r = 0; r < K; r++)
          if (answer.find(tmp[r]) != answer.end()) correct++;
      }
    }
    auto recall = 1.0f * correct / repeat_test / (query_num * K);

    auto time_end = std::chrono::steady_clock::now();
    auto time_us_per_query = (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / (query_num * repeat_test);
    std::cout << string_format("L_search %5d, recall %.4f, time_us_per_query %8d \n", L_search, recall, time_us_per_query);
    if (recall > 1.0)
      break;
  }

  return 0;
}