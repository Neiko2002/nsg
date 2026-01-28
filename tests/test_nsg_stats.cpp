#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

#include <vector>
#include <unordered_set>
#include <filesystem>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

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

static auto read_top_list(const char* fname, size_t& d_out, size_t& n_out)
{
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{})
    {
        std::cerr << "error when accessing top list file" << fname << " size is: " << file_size << " message: " << ec.message() << std::endl;
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open())
    {
        std::cerr << "could not open " << fname << std::endl;
        perror("");
        abort();
    }

    uint32_t dims;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(int));
    assert((dims > 0 && dims < 1000000) || !"unreasonable dimension");
    assert(file_size % ((dims + 1) * 4) == 0 || !"weird file size");
    size_t n = file_size / ((dims + 1) * 4);

    d_out = dims;
    n_out = n;

    auto x = std::make_unique<uint32_t[]>(n * (dims + 1));
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x.get()), n * (dims + 1) * sizeof(uint32_t));
    if (!ifstream) 
        assert(ifstream.gcount() == static_cast<int>(n * (dims + 1)) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) 
        memmove(&x[i * dims], &x[1 + i * (dims + 1)], dims * sizeof(uint32_t));

    ifstream.close();
    return x;
}


static uint32_t compute_reachablity_count(std::vector<std::vector<uint32_t>>& graph) {

    auto graph_size = graph.size();
    uint32_t reachable_count = 0;

    unsigned L = 100; // L_search
    unsigned seed = 1998;
    std::mt19937 rng(seed);
    std::vector<unsigned> init_ids(L);
    efanna2e::GenRandom(rng, init_ids.data(), L, graph_size);
    
    // flood fill from this entrance position
    auto checked_ids = std::vector<bool>(graph_size);
    auto check = std::vector<uint32_t>();

    // start with the first nodes
    for (unsigned s: init_ids) {
        checked_ids[s] = true;
        check.emplace_back(s);
    }
    
    // repeat as long as we have nodes to check
	while(check.size() > 0) {	

        // neighbors which will be checked next round
        auto check_next = std::vector<uint32_t>();

        // get the neighbors to check next
        for (auto &&check_index : check) {
 
            auto& neighbor_indizies = graph[check_index];        
            auto const &neighbors = graph[check_index];

            for (int i = 0; i < neighbor_indizies.size(); i++) {
                auto neighbor_index = neighbor_indizies[i];
                
                if(checked_ids[neighbor_index] == false) {
                    checked_ids[neighbor_index] = true;
                    check_next.emplace_back(neighbor_index);
                }
            }
        }

        check = std::move(check_next);
    }

    // how many nodes have been checked
    uint32_t checked_node_count = 0;
    for (size_t i = 0; i < graph_size; i++)
        if(checked_ids[i])
            checked_node_count++;

    std::cout << "Seed Reachablity " << checked_node_count << " of " << graph_size << " vertices" << std::endl;
    return checked_node_count;
}


struct VertexReach {
  size_t vertex_id;
  uint32_t reach_count;
  std::vector<bool> reachable_ids;

  VertexReach(size_t vertex_id, uint32_t reach_count, std::vector<bool>& reachable_ids) 
        : vertex_id(vertex_id), reach_count(reach_count), reachable_ids(std::move(reachable_ids)) {}
};

static uint32_t compute_avg_reach(std::vector<std::vector<uint32_t>>& graph) {
    const auto graph_size = graph.size();
    auto time_begin = std::chrono::steady_clock::now();

    // remember those vertices which have a very high reach
    uint32_t best_vertex_reach = 0;                             
    auto vertices_reach = std::vector<VertexReach>();    
    auto index_of_vertex_reach = std::vector<uint32_t>(graph_size);    
    std::fill(index_of_vertex_reach.begin(), index_of_vertex_reach.end(), graph_size);

    // find the reach of each vertex
    uint64_t counter = 0;
    uint64_t avg_reach = 0;
    for (size_t entry_id = 0; entry_id < graph_size; entry_id++) {
        
        // flood fill from this entrance position 
        auto checked_ids = std::vector<bool>(graph_size);
        auto check = std::vector<uint32_t>();
        
        // start with the first node
        checked_ids[entry_id] = true;
        check.emplace_back(entry_id);

        // we try to speed up the process by reaching a vertex which can reach a lot of other vertices
        uint32_t best_reach_vertex_index = 0;
        uint32_t best_reach_vertex_reach = 0;
        
        // repeat as long as we have nodes to check
		while(check.size() > 0 && best_reach_vertex_reach < graph_size) {	

            // neighbors which will be checked next round
            auto check_next = std::vector<uint32_t>();

            // get the neighbors to check next
            for (size_t c = 0; c < check.size() && best_reach_vertex_reach < graph_size; c++) {
                const auto check_index = check[c];
                const auto& neighbor_indizies = graph[check_index]; 

                if(neighbor_indizies.size() == 0)
                    std::cout << "zero out-degree for vertex " << check_index << std::endl;

                for (int n = 0; n < neighbor_indizies.size(); n++) {
                    const auto neighbor_index = neighbor_indizies[n];
                    
                    // consider only neighbors which have not been checked yet
                    if(checked_ids[neighbor_index] == false) {
                        checked_ids[neighbor_index] = true;
                        check_next.emplace_back(neighbor_index);

                        // is the neighbor connected to a vertex which can reach a lot of other vertices
                        const auto vertex_reach_index = index_of_vertex_reach[neighbor_index];
                        if(vertex_reach_index < graph_size) {
                            const auto& neighbor_reach = vertices_reach[vertex_reach_index];

                            // found one of the best vertices or a vertex which can reach the best
                            if(neighbor_reach.reach_count > best_reach_vertex_reach) {
                                best_reach_vertex_index = vertex_reach_index;
                                best_reach_vertex_reach = neighbor_reach.reach_count;

                                // found a vertex which can reach all other vertices
                                if(neighbor_reach.reach_count == graph_size) 
                                   break;

                                // copy the reach of the best
                                const auto& best_vertex_checked_ids = neighbor_reach.reachable_ids;
                                for (size_t b = 0; b < graph_size; b++) 
                                    checked_ids[b] = checked_ids[b] | best_vertex_checked_ids[b];
                            }
                        }
                    }
                }
            }

            check = std::move(check_next);
        }

        // found path to a vertex which can reach every other vertex
        if(best_reach_vertex_reach == graph_size) {
            index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            avg_reach += graph_size;

        } else {

            // how many nodes have been checked
            uint32_t reach_count =  0;
            for (size_t i = 0; i < graph_size; i++)
                reach_count += checked_ids[i];
            avg_reach += reach_count;
            
            // is this a new best vertex?
            if(best_vertex_reach < reach_count) {
                best_vertex_reach = reach_count;
                index_of_vertex_reach[entry_id] = (uint32_t) vertices_reach.size();
                vertices_reach.emplace_back(entry_id, reach_count, std::move(checked_ids));
            } else if(best_reach_vertex_reach > 0) {
                index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            } else {
                index_of_vertex_reach[entry_id] = (uint32_t) vertices_reach.size();
                vertices_reach.emplace_back(entry_id, reach_count, std::move(checked_ids));
            }
        }

        counter++;
        if(counter % 10000 == 0) {
            const auto time_sec = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - time_begin).count();
            std::cout << string_format("Avg reach is %.2f after checking %7d of %7d vertices after %4ds\n", ((float)avg_reach)/counter, counter, graph_size, time_sec);
        }
    }  

    const auto time_sec = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - time_begin).count();
    std::cout << string_format("Avg reach is %.2f after checking %7d of %7d vertices after %4ds\n", ((float)avg_reach)/counter, counter, graph_size, time_sec);
    return (uint32_t)(avg_reach/graph_size);
}

static void compute_stats(const char* graph_file, const uint32_t feature_dims, const char* top_list_file) {
    std::cout << "Compute graph stats of " << graph_file << std::endl;


    size_t top_list_dims;
    size_t top_list_count;
    const auto all_top_list = read_top_list(top_list_file, top_list_dims, top_list_count);
    std::cout << "Load TopList from file" << top_list_file << " with " << top_list_count << " elements and k=" << top_list_dims << std::endl;

    auto index = efanna2e::IndexNSG(feature_dims, top_list_count, efanna2e::L2, nullptr);
    index.Load(graph_file);
    auto graph = index.getCompactGraph();
    auto graph_size = graph.size();
    std::cout << "finished getCompactGraph" << std::endl;

    // compute the graph quality
    float perfect_neighbor_ratio = 0;
    float avg_edge_count = 0;
    {
        uint64_t perfect_neighbor_count = 0;
        uint64_t total_neighbor_count = 0;
        for (uint32_t n = 0; n < graph_size; n++) {
            auto& neighbor_indizies = graph[n];
            auto edges_per_node = neighbor_indizies.size();

            // get top list of this node
            auto top_list = all_top_list.get() + n * top_list_dims;
            if(top_list_dims < edges_per_node) {
                std::cerr << "TopList for " << n << " is not long enough has " << edges_per_node << " elements has " << top_list_dims << std::endl;
                edges_per_node = (uint16_t) top_list_dims;
            }
            total_neighbor_count += edges_per_node;

            // check if every neighbor is from the perfect neighborhood
            for (uint32_t e = 0; e < edges_per_node; e++) {
                auto neighbor_index = neighbor_indizies[e];

                // find in the neighbor ini the first few elements of the top list
                for (uint32_t i = 0; i < edges_per_node; i++) {
                    if(neighbor_index == top_list[i]) {
                        perfect_neighbor_count++;
                        break;
                    }
                }
            }
        }

        perfect_neighbor_ratio = ((float) perfect_neighbor_count) / total_neighbor_count;
        avg_edge_count = ((float) total_neighbor_count) / graph_size;
    }
        std::cout << "finished gq" << std::endl;


    // compute the min and max out degree
    uint32_t min_out =  std::numeric_limits<uint32_t>::max();
    uint32_t max_out = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto& neighbor_indizies = graph[n];
        auto edges_per_node = neighbor_indizies.size();

        if(edges_per_node < min_out)
            min_out = edges_per_node;
        if(max_out < edges_per_node)
            max_out = edges_per_node;
    }
        std::cout << "finished out" << std::endl;

      // compute the in_degree per vertex
    auto in_degree_count = std::vector<uint32_t>(graph_size);
    for (uint32_t n = 0; n < graph_size; n++) {
        auto& neighbor_indizies = graph[n];
        auto edges_per_node = neighbor_indizies.size();

        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = neighbor_indizies[e];
            in_degree_count[neighbor_index]++;
        }
    }
        std::cout << "finished indegree count" << std::endl;


    // compute the min and max in degree
    uint32_t min_in = std::numeric_limits<uint32_t>::max();
    uint32_t max_in = 0;
    uint32_t source_nodes = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto in_degree = in_degree_count[n];

        if(in_degree < min_in)
            min_in = in_degree;
        if(max_in < in_degree)
            max_in = in_degree;
        if(in_degree == 0) 
            source_nodes++;
    }
    std::cout << "finished calcs" << std::endl;
    std::printf("GQ %.4f, avg degree %.1f, min_out %d, max_out %d, min_in %d, max_in %d, source vertices %d, vertex count %zd\n", perfect_neighbor_ratio, avg_edge_count, min_out, max_out, min_in, max_in, source_nodes, graph_size);


    auto reachability_count = compute_reachablity_count(graph);
    auto avg_reach = compute_avg_reach(graph);
    std::printf("search reachability count %d, exploration avg reach %d\n", reachability_count, avg_reach);
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

    // // ------------------------------------------ SIFT1M --------------------------------------------------
    // const auto dims = 128;
    // const auto top_list_file  = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base_top1000.ivecs)";
    // // GQ 0.3876, avg degree 29.8, min_out 1, max_out 50, min_in 1, max_in 118, source nodes 0, search reachability count 1000000, exploration avg reach 1000000.00, node count 1000000
    // // const auto nsg_file    = R"(e:/Data/Feature/SIFT1M/nsg/sift.nsg)";
    // const auto nsg_file       = R"(e:/Data/Feature/SIFT1M/nsg/sift_L40_R50_C500_efaK200_L200_It10_S10_R100.nsg)";


    // ------------------------------------------ GloVe --------------------------------------------------
    // const auto dims = 100;
    // const auto top_list_file  = R"(e:/Data/Feature/GloVe/glove-100/glove-100_base_top1000.ivecs)";

    // // GQ 0.3230, avg degree 13.3, min_out 1, max_out 83, min_in 1, max_in 936, source nodes 0, search reachability count 1183514, exploration avg reach 1183514.00, node count 1183514
    // // auto nsg_file       = R"(c:/Data/Feature/GloVe/nsg/glove-100_L40_R50_C500_efa20It.nsg)";

    // // GQ 0.3194, avg degree 13.7, min_out 1, max_out 76, min_in 1, max_in 934, source nodes 0, search reachability count 1183514, exploration avg reach 1183514.00, node count 1183514
    // // auto nsg_file       = R"(c:/Data/Feature/GloVe/nsg/glove-100_L400_R50_C500_efa20It.nsg)";

    // auto nsg_file       = R"(e:/Data/Feature/GloVe/nsg/glove-100_L50_R70_C500_EfaK400_L420_It12_S15_R200.nsg)";


    // ------------------------------------------ Enron --------------------------------------------------
    // const auto dims = 1368;
    // const auto top_list_file  = R"(e:/Data/Feature/Enron/enron/enron_base_top1000.ivecs)";
    // const auto nsg_file       = R"(e:/Data/Feature/Enron/nsg/L150_R60_C600_efaK200_L200_It7_S25_R200.nsg)";


    // ------------------------------------------ Audio --------------------------------------------------
    const auto dims = 192;
    const auto top_list_file  = R"(e:/Data/Feature/Audio/audio/audio_base_top1000.ivecs)";
    const auto nsg_file       = R"(e:/Data/Feature/Audio/nsg/L200_R30_C600_efaK200_L230_It5_S10_R100.nsg)";

    compute_stats(nsg_file, dims, top_list_file);

    std::cout << "Finished" << std::endl;
}