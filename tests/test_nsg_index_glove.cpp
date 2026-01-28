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

  auto object_file      = R"(e:/Data/Feature/GloVe/glove-100/glove-100_base.fvecs)";
  auto nsg_file         = R"(e:/Data/Feature/GloVe/nsg/glove-100_L50_R70_C500_EfaK400_L420_It12_S15_R200.nsg)";
  auto efanna_file      = std::string("e:/Data/Feature/GloVe/efanna/glove-100_K400_L420_It12_S15_R200.efa");
  //auto efanna_file      = std::string("e:/Data/Feature/GloVe/efanna/glove-100_400nn.knng");

  // load feature vectors
  std::cout << "Load basedata and align" << std::endl;
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(object_file, data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim); // align the data before build
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading base data and align" << std::endl;

  // parameters of the SSG paper
  unsigned L = 50;
  unsigned R = 70;
  unsigned C = 500;  // default on all datasets in the NSG paper

  // create the index
  std::cout << "create graph" << std::endl;
  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after creating graph" << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<std::string>("nn_graph_path", efanna_file);

  std::cout << "build graph using " << efanna_file << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "indexing time: " << diff.count() << "\n";
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after building graph" << std::endl;

  index.Save(nsg_file);

  return 0;
}
