#pragma once
#include <vector>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cstdint>



struct LegoBrick {
    int id;
    int x, y, z;
    int sx, sy;
    int w;
    double mass;
    bool alive;

    LegoBrick() : id(-1), x(0), y(0), z(0), sx(1), sy(1), w(0), mass(0.0), alive(true) {}
    LegoBrick(int _id, int _x, int _y, int _z, int _sx, int _sy, double _mass = 0.0)
        : id(_id), x(_x), y(_y), z(_z), sx(_sx), sy(_sy), w(0), mass(_mass), alive(true) {}

    inline std::pair<int,int> xspan() const { return {x, x + sx}; }
    inline std::pair<int,int> yspan() const { return {y, y + sy}; }

    static bool has_valid_size(int sx, int sy) {
        int a = std::min(sx, sy);
        int b = std::max(sx, sy);
        if (a == 1 && (b == 1 || b == 2 || b == 3 || b == 4 || b == 6 || b == 8)) return true;
        if (a == 2 && (b == 2 || b == 3 || b == 4 || b == 6 || b == 8 || b == 10 )) return true;
        return false;
    }
};

struct SolveResult {
    int components = 0;
    std::vector<std::array<int,5>> bricks;
};

class LegoBlockGraph {
public:
    int K_N = 2;
    int F_MAX = 200;

    std::vector<LegoBrick> bricks_;
    std::vector<std::vector<int>> graph_;
    std::vector<std::vector<int>> reverse_graph_;
    std::vector<std::vector<int>> horizontal_neigh_;
    std::unordered_map<int, std::vector<int>> levels_;
    std::vector<int> visited_;
    std::mt19937 rng_;

    double get_mass_from_library(int sx, int sy) const;



public:
    explicit LegoBlockGraph(uint32_t seed = std::random_device{}());
    static LegoBlockGraph from_occ_u8(const uint8_t* occ, int X, int Y, int Z, uint32_t seed = std::random_device{}());

    int add_block(int x, int y, int z, int sx, int sy);
    void remove_block(int id);

    void merge_to_maximal(int preferred_axis = -1);
    int generate_single_component_analysis(int max_iter = 100);

    SolveResult export_result() const;
    LegoBlockGraph clone() const;

private:
    static bool overlap_1d(const std::pair<int,int>& a, const std::pair<int,int>& b);
    static bool are_connected(const LegoBrick& a, const LegoBrick& b);
    static bool are_adjacent(const LegoBrick& a, const LegoBrick& b);
    static bool can_merge(const LegoBrick& a, const LegoBrick& b);
    static LegoBrick merge_bricks(const LegoBrick& a, const LegoBrick& b, int new_id);

    std::vector<int> filter_connected(const std::vector<int>& ids, int ref_id) const;
    std::vector<int> get_horizontal_neighbours_from_all(int ref_id) const;
    std::vector<int> get_neighbours_all(int id) const;

    std::vector<std::pair<int,int>> generate_mergables() const;
    std::pair<int,int> component_analysis();
    std::unordered_set<int> k_neighbourhood(int root_id, int k) const;
    std::vector<int> split_to_unit(int id, std::vector<std::array<int,5>>& unit_specs) const;
    LegoBlockGraph shuffle(int w_id);

    bool is_alive(int id) const {
        return id >= 0 && id < (int)bricks_.size() && bricks_[id].alive;
    }

    void ensure_adj_size(int n);
    std::vector<int> voxel_grid_;
    inline int pack_coords(int x, int y, int z) const {
        if (x < 0 || x >= max_X_ || y < 0 || y >= max_Y_ || z < 0 || z >= max_Z_) return -1;
        return (x * max_Y_ + y) * max_Z_ + z;
    }
    int count_gaps(int x, int y, int z, int sx, int sy) const;
    int max_X_ = 0, max_Y_ = 0, max_Z_ = 0;
    void merge_layer_to_maximal(int z);
};