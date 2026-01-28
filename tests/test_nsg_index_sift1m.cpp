//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

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

  // // ------------------------------------ SIFT1M -----------------------------------------
  // auto object_file      = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  // auto nsg_file         = R"(e:/Data/Feature/SIFT1M/nsg/sift_L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";
  // auto efanna_file      = R"(e:/Data/Feature/SIFT1M/efanna/efanna_K200_L200_It10_S10_R100.efa)"; //  no spaces

  // // https://github.com/Neiko2002/nsg#parameters-used-in-our-paper
  // unsigned L = 40;
  // unsigned R = 50;
  // unsigned C = 500;

  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // // unsigned L = 150;
  // // unsigned R = 30;
  // // unsigned C = 400;



  // // ------------------------------------ Enron -----------------------------------------
  // auto object_file      = R"(e:/Data/Feature/Enron/enron/enron_base.fvecs)";
  // auto nsg_file         = R"(e:/Data/Feature/Enron/nsg/L150_R60_C600_efaK200_L200_It7_S25_R200.nsg)";
  // auto efanna_file      = R"(e:/Data/Feature/Enron/efanna/K200_L200_It7_S25_R200.efa)"; //  no spaces

  // // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // unsigned L = 150;
  // unsigned R = 60;
  // unsigned C = 600;



  // ------------------------------------ Audio -----------------------------------------
  // auto object_file      = R"(e:/Data/Feature/Audio/audio/audio_base.fvecs)";
  // auto nsg_file         = R"(e:/Data/Feature/Audio/nsg/L200_R30_C600_efaK200_L230_It5_S10_R100.nsg)";
  // auto efanna_file      = R"(e:/Data/Feature/Audio/efanna/K200_L230_It5_S10_R100_(nTrees0_mLevel0).efa)"; //  no spaces

  // // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // unsigned L = 200;
  // unsigned R = 30;
  // unsigned C = 600;


  // // ------------------------------------ ImageNet1k -----------------------------------------
  // auto object_file      = R"(e:/Data/Feature/ImageNet1k/ImageNet1k/clip_base.fvecs)";
  // auto nsg_file         = R"(e:/Data/Feature/ImageNet1k/nsg/sift_L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";
  // auto efanna_file      = R"(e:/Data/Feature/ImageNet1k/efanna/K200_L200_It10_S10_R100_(nTrees0_mLevel0).efa)"; //  no spaces

  // // NSGs parameter for Sift1M
  // // https://github.com/Neiko2002/nsg#parameters-used-in-our-paper
  // unsigned L = 40;
  // unsigned R = 50;
  // unsigned C = 500;


  // // ------------------------------------ Deep10M -----------------------------------------
  // auto object_file      = R"(e:/Data/Feature/Deep10M/deep10m/deep10m_base.fvecs)";
  // auto nsg_file         = R"(e:/Data/Feature/Deep10M/nsg/deep10m_L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";
  // auto efanna_file      = R"(e:/Data/Feature/Deep10M/efanna/K200_L200_It10_S10_R100_(nTrees0_mLevel0).efa)"; //  no spaces

  // // NSGs parameter for Sift1M
  // // https://github.com/Neiko2002/nsg#parameters-used-in-our-paper
  // unsigned L = 40;
  // unsigned R = 50;
  // unsigned C = 500;

    // ------------------------------------ Deep1M -----------------------------------------
  auto object_file      = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_base.fvecs)";
  auto nsg_file         = R"(e:/Data/Feature/Deep1M/nsg/L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";
  auto efanna_file      = R"(e:/Data/Feature/Deep1M/efanna/K200_L200_It10_S10_R100_(nTrees0_mLevel0).efa)"; //  no spaces

  // NSGs parameter for Sift1M
  unsigned L = 40;
  unsigned R = 50;
  unsigned C = 500;


  // ---------------------------------------------------------------------------
  // -------------------------------- NSG --------------------------------------
  // ---------------------------------------------------------------------------
  // load feature vectors
  std::cout << "Load basedata and align" << std::endl;
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(object_file, data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim); // align the data before build
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading base data and align" << std::endl;

  // create the index
  std::cout << "create graph" << std::endl;
  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after creating graph" << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<std::string>("nn_graph_path", efanna_file);

  std::cout << "build graph" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "indexing time: " << diff.count() << "\n";
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after building graph" << std::endl;

  index.Save(nsg_file);

  return 0;
}
