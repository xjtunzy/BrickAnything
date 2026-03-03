#include "lego_block_graph.hpp"
#include <set>
#include <iostream>



LegoBlockGraph::LegoBlockGraph(uint32_t seed) : rng_(seed) {}

void LegoBlockGraph::ensure_adj_size(int n) {
    if ((int)graph_.size() < n) graph_.resize(n);
    if ((int)reverse_graph_.size() < n) reverse_graph_.resize(n);
    if ((int)horizontal_neigh_.size() < n) horizontal_neigh_.resize(n);
    if ((int)visited_.size() < n) visited_.resize(n, -1);
}

bool LegoBlockGraph::overlap_1d(const std::pair<int,int>& a, const std::pair<int,int>& b) {
    // 半开区间 [l, r)
    return !(a.second <= b.first || b.second <= a.first);
}

bool LegoBlockGraph::are_connected(const LegoBrick& a, const LegoBrick& b) {
    if (std::abs(a.z - b.z) != 1) return false;
    return overlap_1d(a.xspan(), b.xspan()) && overlap_1d(a.yspan(), b.yspan());
}

bool LegoBlockGraph::are_adjacent(const LegoBrick& a, const LegoBrick& b) {
    if (a.z != b.z) return false;
    bool x_touch = (a.x + a.sx == b.x) || (b.x + b.sx == a.x);
    bool y_ovl   = overlap_1d(a.yspan(), b.yspan());
    bool y_touch = (a.y + a.sy == b.y) || (b.y + b.sy == a.y);
    bool x_ovl   = overlap_1d(a.xspan(), b.xspan());
    return (x_touch && y_ovl) || (y_touch && x_ovl);
}

bool LegoBlockGraph::can_merge(const LegoBrick& a, const LegoBrick& b) {
    if (a.z != b.z) return false;

    // same y-span => merge along x
    if (a.y == b.y && a.sy == b.sy) {
        int left_id  = (a.x <= b.x) ? 0 : 1;
        const LegoBrick& L = (left_id == 0) ? a : b;
        const LegoBrick& R = (left_id == 0) ? b : a;
        if (L.x + L.sx == R.x) {
            int sx = a.sx + b.sx;
            if (LegoBrick::has_valid_size(sx, a.sy)) return true;
        }
    }

    // same x-span => merge along y
    if (a.x == b.x && a.sx == b.sx) {
        int low_id = (a.y <= b.y) ? 0 : 1;
        const LegoBrick& L = (low_id == 0) ? a : b;
        const LegoBrick& R = (low_id == 0) ? b : a;
        if (L.y + L.sy == R.y) {
            int sy = a.sy + b.sy;
            if (LegoBrick::has_valid_size(a.sx, sy)) return true;
        }
    }

    return false;
}

LegoBrick LegoBlockGraph::merge_bricks(const LegoBrick& a, const LegoBrick& b, int new_id) {
    if (a.z != b.z) throw std::runtime_error("merge_bricks z mismatch");

    // X merge
    if (a.y == b.y && a.sy == b.sy && ((a.x + a.sx == b.x) || (b.x + b.sx == a.x))) {
        int x = std::min(a.x, b.x);
        return LegoBrick(new_id, x, a.y, a.z, a.sx + b.sx, a.sy);
    }
    // Y merge
    if (a.x == b.x && a.sx == b.sx && ((a.y + a.sy == b.y) || (b.y + b.sy == a.y))) {
        int y = std::min(a.y, b.y);
        return LegoBrick(new_id, a.x, y, a.z, a.sx, a.sy + b.sy);
    }

    throw std::runtime_error("merge called on non-mergeable pair");
}

std::vector<int> LegoBlockGraph::filter_connected(const std::vector<int>& ids, int ref_id) const {
    std::vector<int> out;
    const LegoBrick& ref = bricks_[ref_id];
    for (int id : ids) {
        if (!is_alive(id)) continue;
        if (are_connected(bricks_[id], ref)) out.push_back(id);
    }
    return out;
}

std::vector<int> LegoBlockGraph::get_horizontal_neighbours_from_all(int ref_id) const {
    const LegoBrick& b = bricks_[ref_id];
    std::unordered_set<int> unique_neighbors;

    auto check_and_add = [&](int x, int y, int z) {
        int idx = pack_coords(x, y, z); // 使用你新写的 pack_coords
        if (idx != -1) {
            int nid = voxel_grid_[idx]; // 直接读数组
            if (nid != -1 && nid != ref_id) unique_neighbors.insert(nid);
        }
    };

    // 检查四周边界（X/Y方向）
    for (int dy = 0; dy < b.sy; ++dy) {
        check_and_add(b.x - 1, b.y + dy, b.z);
        check_and_add(b.x + b.sx, b.y + dy, b.z);
    }
    for (int dx = 0; dx < b.sx; ++dx) {
        check_and_add(b.x + dx, b.y - 1, b.z);
        check_and_add(b.x + dx, b.y + b.sy, b.z);
    }

    return std::vector<int>(unique_neighbors.begin(), unique_neighbors.end());
}

std::vector<int> LegoBlockGraph::get_neighbours_all(int id) const {
    std::vector<int> out;
    if (!is_alive(id)) return out;

    std::unordered_set<int> seen;
    for (int v : graph_[id]) if (is_alive(v) && !seen.count(v)) { out.push_back(v); seen.insert(v); }
    for (int v : reverse_graph_[id]) if (is_alive(v) && !seen.count(v)) { out.push_back(v); seen.insert(v); }
    for (int v : horizontal_neigh_[id]) if (is_alive(v) && !seen.count(v)) { out.push_back(v); seen.insert(v); }
    return out;
}

// 实现质量查找函数
double LegoBlockGraph::get_mass_from_library(int sx, int sy) const {
    int a = std::min(sx, sy);
    int b = std::max(sx, sy);
    if (a == 2 && b == 4)  return 0.00216;
    if (a == 2 && b == 6)  return 0.00323;
    if (a == 1 && b == 8)  return 0.00303;
    if (a == 1 && b == 4)  return 0.00157;
    if (a == 1 && b == 6)  return 0.00228;
    if (a == 1 && b == 2)  return 0.00081;
    if (a == 1 && b == 1)  return 0.00043;
    if (a == 2 && b == 2)  return 0.00115;
    if (a == 2 && b == 3)  return 0.00115;
    if (a == 2 && b == 8)  return 0.00115;
    if (a == 2 && b == 10) return 0.00115;
    if (a == 1 && b == 3)  return 0.00115;
    return 0.0;
}

int LegoBlockGraph::add_block(int x, int y, int z, int sx, int sy) {
    int id = static_cast<int>(bricks_.size());
    double mass = get_mass_from_library(sx, sy);
    bricks_.emplace_back(id, x, y, z, sx, sy, mass);
    ensure_adj_size(id + 1);

    levels_[z].push_back(id);

    // 1. 更新 voxel_grid_ (索引由索引函数 pack_coords 提供)
    for (int dx = 0; dx < sx; ++dx) {
        for (int dy = 0; dy < sy; ++dy) {
            int idx = pack_coords(x + dx, y + dy, z);
            if (idx != -1) voxel_grid_[idx] = id;
        }
    }

    // 2. 更新上下层连接关系 (这部分逻辑保持不变，但内部会用到高效的邻居查找)
    auto it_up = levels_.find(z + 1);
    if (it_up != levels_.end()) {
        graph_[id] = filter_connected(it_up->second, id);
        for (int nb : graph_[id]) reverse_graph_[nb].push_back(id);
    }

    auto it_dn = levels_.find(z - 1);
    if (it_dn != levels_.end()) {
        reverse_graph_[id] = filter_connected(it_dn->second, id);
        for (int nb : reverse_graph_[id]) graph_[nb].push_back(id);
    }

    // 3. 更新同层邻居 (这里会调用我们优化过的 get_horizontal_neighbours_from_all)
    horizontal_neigh_[id] = get_horizontal_neighbours_from_all(id);
    for (int nb : horizontal_neigh_[id]) {
        horizontal_neigh_[nb].push_back(id);
    }

    return id;
}

void LegoBlockGraph::remove_block(int id) {
    if (!is_alive(id)) return;
    LegoBrick& b = bricks_[id];

    // 1. 从索引数组中抹除
    for (int dx = 0; dx < b.sx; ++dx) {
        for (int dy = 0; dy < b.sy; ++dy) {
            int idx = pack_coords(b.x + dx, b.y + dy, b.z);
            if (idx != -1) voxel_grid_[idx] = -1;
        }
    }

    // 2. 从层级记录中移除
    auto it = levels_.find(b.z);
    if (it != levels_.end()) {
        auto& vec = it->second;
        vec.erase(std::remove(vec.begin(), vec.end(), id), vec.end());
    }

    // 3. 清理图连接关系 (保持原样)
    for (int ch : graph_[id]) {
        if (!is_alive(ch)) continue;
        auto& rv = reverse_graph_[ch];
        rv.erase(std::remove(rv.begin(), rv.end(), id), rv.end());
    }
    for (int pa : reverse_graph_[id]) {
        if (!is_alive(pa)) continue;
        auto& gv = graph_[pa];
        gv.erase(std::remove(gv.begin(), gv.end(), id), gv.end());
    }
    for (int nb : horizontal_neigh_[id]) {
        if (!is_alive(nb)) continue;
        auto& hv = horizontal_neigh_[nb];
        hv.erase(std::remove(hv.begin(), hv.end(), id), hv.end());
    }

    graph_[id].clear();
    reverse_graph_[id].clear();
    horizontal_neigh_[id].clear();
    b.alive = false;
}


int LegoBlockGraph::count_gaps(int x, int y, int z, int sx, int sy) const {
    if (z <= 0) return 0;
    int total_gaps = 0;
    int z_under = z - 1;

    // 检查 X 方向跨缝
    for (int dy = 0; dy < sy; ++dy) {
        for (int dx = 0; dx < sx - 1; ++dx) {
            int id1 = voxel_grid_[pack_coords(x + dx, y + dy, z_under)];
            int id2 = voxel_grid_[pack_coords(x + dx + 1, y + dy, z_under)];
            if (id1 != -1 && id2 != -1 && id1 != id2) total_gaps++;
        }
    }
    // 检查 Y 方向跨缝
    for (int dx = 0; dx < sx; ++dx) {
        for (int dy = 0; dy < sy - 1; ++dy) {
            int id1 = voxel_grid_[pack_coords(x + dx, y + dy, z_under)];
            int id2 = voxel_grid_[pack_coords(x + dx, y + dy + 1, z_under)];
            if (id1 != -1 && id2 != -1 && id1 != id2) total_gaps++;
        }
    }
    return total_gaps;
}

LegoBlockGraph LegoBlockGraph::from_occ_u8(const uint8_t* occ, int X, int Y, int Z, uint32_t seed) {
    LegoBlockGraph g(seed);
    g.max_X_ = X; g.max_Y_ = Y; g.max_Z_ = Z;
    g.voxel_grid_.assign(X * Y * Z, -1);
    auto idx = [Y, Z](int x, int y, int z) {
        return x * Y * Z + y * Z + z;
    };

    for (int z = 0; z < Z; ++z) {
        // 1. 先把这一层所有的 1x1 块添加进去
        std::vector<int> current_layer_ids;
        for (int y = 0; y < Y; ++y) {
            for (int x = 0; x < X; ++x) {
                if (occ[idx(x,y,z)] > 0) {
                    int id = g.add_block(x, y, z, 1, 1);
                    current_layer_ids.push_back(id);
                }
            }
        }
        
        // 2. 立即对这一层执行合并
        // 此时 count_gaps 查找 z-1 层时，z-1 层已经是合并好的大块了
        g.merge_layer_to_maximal(z); 
    }
    return g;
}

void LegoBlockGraph::merge_layer_to_maximal(int z) {
    std::vector<std::pair<int,int>> layer_mergables;
    // 优化点：只从当前 z 层的积木出发寻找候选对
    if (levels_.count(z)) {
        for (int aid : levels_[z]) {
            if (!is_alive(aid)) continue;
            for (int bid : horizontal_neigh_[aid]) {
                // 仅处理同层、未处理过(bid > aid)、且符合物理尺寸的
                if (is_alive(bid) && bricks_[bid].z == z && bid > aid) {
                    if (can_merge(bricks_[aid], bricks_[bid])) {
                        layer_mergables.push_back({aid, bid});
                    }
                }
            }
        }
    }

    while (!layer_mergables.empty()) {
        int best_idx = -1;
        float max_score = -1.0f;

        // 在 lego_block_graph.cpp 中修改此循环
        for (int i = 0; i < (int)layer_mergables.size(); ++i) {
            auto [a, b] = layer_mergables[i];
            if (!is_alive(a) || !is_alive(b) || !can_merge(bricks_[a], bricks_[b])) continue;

            // 1. 获取合并后的几何参数
            int nx = std::min(bricks_[a].x, bricks_[b].x);
            int ny = std::min(bricks_[a].y, bricks_[b].y);
            int nsx = (bricks_[a].y == bricks_[b].y) ? (bricks_[a].sx + bricks_[b].sx) : bricks_[a].sx;
            int nsy = (bricks_[a].x == bricks_[b].x) ? (bricks_[a].sy + bricks_[b].sy) : bricks_[a].sy;

            // 2. 物理支撑分析
            float support_area = 0;
            bool a_has_support = false;
            bool b_has_support = false;

            if (z > 0) {
                // 检查 A 下方是否有支撑
                for(int dx=0; dx<bricks_[a].sx; ++dx) {
                    for(int dy=0; dy<bricks_[a].sy; ++dy) {
                        if(voxel_grid_[pack_coords(bricks_[a].x+dx, bricks_[a].y+dy, z-1)] != -1) {
                            a_has_support = true; break;
                        }
                    }
                    if(a_has_support) break;
                }
                // 检查 B 下方是否有支撑
                for(int dx=0; dx<bricks_[b].sx; ++dx) {
                    for(int dy=0; dy<bricks_[b].sy; ++dy) {
                        if(voxel_grid_[pack_coords(bricks_[b].x+dx, bricks_[b].y+dy, z-1)] != -1) {
                            b_has_support = true; break;
                        }
                    }
                    if(b_has_support) break;
                }
                // 计算合并后总支撑面积
                for (int dx = 0; dx < nsx; ++dx) {
                    for (int dy = 0; dy < nsy; ++dy) {
                        int idx = pack_coords(nx + dx, ny + dy, z - 1);
                        if (idx != -1 && voxel_grid_[idx] != -1) support_area += 1.0f;
                    }
                }
            } else {
                // Z=0 层默认全支撑
                support_area = (float)(nsx * nsy);
                a_has_support = b_has_support = true;
            }

            // 3. 计算最终得分
            // 桥接奖励：如果一个是悬空的，一个是实心的，赋予极高优先级促使合并
            float bridge_bonus = (a_has_support != b_has_support) ? 5000.0f : 0.0f;
            
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float score = (float)count_gaps(nx, ny, z, nsx, nsy) * 2000.0f  // 跨缝权重（最高）
                        + support_area * 100.0f                             // 支撑权重（解决悬空）
                        + bridge_bonus                                      // 桥接权重（强行拉拢碎块）
                        + (float)(nsx * nsy) * 10.0f                        // 面积权重
                        + dist(rng_);
            
            if (score > max_score) {
                max_score = score;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        auto [a_id, b_id] = layer_mergables[best_idx];
        LegoBrick newb = merge_bricks(bricks_[a_id], bricks_[b_id], (int)bricks_.size());

        layer_mergables.erase(layer_mergables.begin() + best_idx);
        
        remove_block(a_id);
        remove_block(b_id);
        int nid = add_block(newb.x, newb.y, newb.z, newb.sx, newb.sy);

        // 只把该层受影响的新邻居加回来
        for (int nb : horizontal_neigh_[nid]) {
            if (is_alive(nb) && bricks_[nb].z == z && can_merge(bricks_[nid], bricks_[nb])) {
                layer_mergables.emplace_back(nid, nb);
            }
        }
    }
}

std::vector<std::pair<int,int>> LegoBlockGraph::generate_mergables() const {
    std::vector<std::pair<int,int>> pairs;
    std::unordered_set<uint64_t> seen;

    for (const auto& a : bricks_) {
        if (!a.alive) continue;
        int aid = a.id;
        for (int bid : horizontal_neigh_[aid]) {
            if (!is_alive(bid)) continue;
            int u = std::min(aid, bid), v = std::max(aid, bid);
            uint64_t key = (uint64_t)u << 32 | (uint32_t)v;
            if (seen.count(key)) continue;
            if (can_merge(bricks_[aid], bricks_[bid])) {
                pairs.emplace_back(aid, bid);
                seen.insert(key);
            }
        }
    }
    return pairs;
}

void LegoBlockGraph::merge_to_maximal(int preferred_axis) {
    // ==========================================================
    // 1. [Probe & Bridge] 探测与锁定机制
    // 在大规模合并前，优先处理那些下方悬空的 1x1 碎块，寻找基座并锁定
    // ==========================================================
    std::vector<int> alive_ids;
    for (int i = 0; i < (int)bricks_.size(); ++i) if (is_alive(i)) alive_ids.push_back(i);

    for (int id : alive_ids) {
        if (!is_alive(id)) continue;
        const auto& b = bricks_[id];
        if (b.z == 0) continue; // 地面层不需要探测

        // 检查是否悬空 (下方完全无支撑)
        bool has_any_support = false;
        for (int dx = 0; dx < b.sx; ++dx) {
            for (int dy = 0; dy < b.sy; ++dy) {
                int idx_below = pack_coords(b.x + dx, b.y + dy, b.z - 1);
                if (idx_below != -1 && voxel_grid_[idx_below] != -1) {
                    has_any_support = true; break;
                }
            }
            if (has_any_support) break;
        }

        if (!has_any_support) {
            // 发现悬空！尝试向四周探测基座 (探测距离最大为 6，匹配乐高长砖)
            int best_target_id = -1;
            int best_dist = 999;
            
            // 四个方向：+X, -X, +Y, -Y
            const int dxs[] = {1, -1, 0, 0};
            const int dys[] = {0, 0, 1, -1};

            for (int d = 0; d < 4; ++d) {
                for (int step = 1; step <= 6; ++step) {
                    int nx = b.x + dxs[d] * step;
                    int ny = b.y + dys[d] * step;
                    int idx = pack_coords(nx, ny, b.z);
                    int idx_below = pack_coords(nx, ny, b.z - 1);

                    if (idx == -1) break; // 出界
                    int target_id = voxel_grid_[idx];
                    if (target_id == -1 || target_id == id) continue; // 空位或自己

                    // 检查该位置下方是否有基座
                    if (idx_below != -1 && voxel_grid_[idx_below] != -1) {
                        // 发现基座！尝试与该方向上的邻居合并
                        if (can_merge(bricks_[id], bricks_[target_id])) {
                            best_target_id = target_id;
                            best_dist = step;
                            break; 
                        }
                    }
                }
                if (best_target_id != -1) break;
            }

            // 锁定并合并：如果找到了基座，立即执行强制合并
            if (best_target_id != -1) {
                LegoBrick newb = merge_bricks(bricks_[id], bricks_[best_target_id], (int)bricks_.size());
                remove_block(id);
                remove_block(best_target_id);
                add_block(newb.x, newb.y, newb.z, newb.sx, newb.sy);
            }
        }
    }

    // ==========================================================
    // 2. [Greedy Maximal Merge] 常规贪心合并
    // 基于跨缝、桥接分和轴向偏好进行全局优化
    // ==========================================================
    auto mergables = generate_mergables();
    
    while (!mergables.empty()) {
        int best_idx = -1;
        float max_score = -1.0f;

        for (int i = 0; i < (int)mergables.size(); ++i) {
            auto [a, b] = mergables[i];
            if (!is_alive(a) || !is_alive(b) || !can_merge(bricks_[a], bricks_[b])) continue;

            int nx = std::min(bricks_[a].x, bricks_[b].x);
            int ny = std::min(bricks_[a].y, bricks_[b].y);
            int z = bricks_[a].z;
            int nsx = (bricks_[a].y == bricks_[b].y) ? (bricks_[a].sx + bricks_[b].sx) : bricks_[a].sx;
            int nsy = (bricks_[a].x == bricks_[b].x) ? (bricks_[a].sy + bricks_[b].sy) : bricks_[a].sy;

            float support_area = 0;
            bool a_has_support = false;
            bool b_has_support = false;

            if (z > 0) {
                for(int dx=0; dx<bricks_[a].sx; ++dx) {
                    for(int dy=0; dy<bricks_[a].sy; ++dy) {
                        int idx = pack_coords(bricks_[a].x+dx, bricks_[a].y+dy, z-1);
                        if(idx != -1 && voxel_grid_[idx] != -1) { a_has_support = true; break; }
                    }
                    if(a_has_support) break;
                }
                for(int dx=0; dx<bricks_[b].sx; ++dx) {
                    for(int dy=0; dy<bricks_[b].sy; ++dy) {
                        int idx = pack_coords(bricks_[b].x+dx, bricks_[b].y+dy, z-1);
                        if(idx != -1 && voxel_grid_[idx] != -1) { b_has_support = true; break; }
                    }
                    if(b_has_support) break;
                }
                for (int dx = 0; dx < nsx; ++dx) {
                    for (int dy = 0; dy < nsy; ++dy) {
                        int idx = pack_coords(nx + dx, ny + dy, z - 1);
                        if (idx != -1 && voxel_grid_[idx] != -1) support_area += 1.0f;
                    }
                }
            } else {
                support_area = (float)(nsx * nsy);
                a_has_support = b_has_support = true;
            }

            // 轴向偏好分：大幅提升权重以应对细长模型 (从 800 提升至 5000)
            float axis_bonus = 0;
            float direction_weight = 5000.0f; 
            if (preferred_axis == 0 && bricks_[a].y == bricks_[b].y) axis_bonus = direction_weight; 
            if (preferred_axis == 1 && bricks_[a].x == bricks_[b].x) axis_bonus = direction_weight;

            // 桥接奖励：提升权重以确保悬空部分被吸纳 (从 5000 提升至 10000)
            float bridge_bonus = (a_has_support != b_has_support) ? 10000.0f : 0.0f;
            
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            float score = (float)count_gaps(nx, ny, z, nsx, nsy) * 2000.0f 
                        + bridge_bonus 
                        + support_area * 100.0f
                        + (float)(nsx * nsy) * 10.0f
                        + axis_bonus 
                        + dist(rng_);
            
            if (score > max_score) {
                max_score = score;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        auto [a_id, b_id] = mergables[best_idx];
        LegoBrick newb = merge_bricks(bricks_[a_id], bricks_[b_id], (int)bricks_.size());

        mergables.erase(mergables.begin() + best_idx);
        
        remove_block(a_id);
        remove_block(b_id);
        int nid = add_block(newb.x, newb.y, newb.z, newb.sx, newb.sy);

        for (int nb : horizontal_neigh_[nid]) {
            if (is_alive(nb) && bricks_[nb].z == newb.z && can_merge(bricks_[nid], bricks_[nb])) {
                mergables.emplace_back(nid, nb);
            }
        }
    }
}

std::pair<int,int> LegoBlockGraph::component_analysis() {
    // init visited
    visited_.assign(bricks_.size(), -1);

    int A = 0;
    for (const auto& bi : bricks_) {
        if (!bi.alive) continue;
        if (visited_[bi.id] != -1) continue;

        std::deque<int> q;
        q.push_back(bi.id);
        visited_[bi.id] = A;

        while (!q.empty()) {
            int u = q.front(); q.pop_front();
            for (int v : graph_[u]) {
                if (!is_alive(v)) continue;
                if (visited_[v] == -1) {
                    visited_[v] = A;
                    q.push_back(v);
                }
            }
            for (int v : reverse_graph_[u]) {
                if (!is_alive(v)) continue;
                if (visited_[v] == -1) {
                    visited_[v] = A;
                    q.push_back(v);
                }
            }
        }
        ++A;
    }

    std::vector<int> concerned;
    int total_w = 0;

    for (auto& b : bricks_) {
        if (!b.alive) continue;
        std::unordered_set<int> comps;
        comps.insert(visited_[b.id]);

        for (int nb : get_neighbours_all(b.id)) {
            comps.insert(visited_[nb]);
        }

        b.w = (int)comps.size() - 1;
        total_w += b.w;
        if (b.w > 0) concerned.push_back(b.id);
    }

    // if no problematic brick
    if (total_w == 0) {
        int any_id = -1;
        for (const auto& b : bricks_) {
            if (b.alive) { any_id = b.id; break; }
        }
        return {A, any_id};
    }

    std::uniform_int_distribution<int> dist(0, total_w - 1);
    int pick = dist(rng_);
    int acc = 0;
    for (int id : concerned) {
        acc += bricks_[id].w;
        if (acc > pick) return {A, id};
    }
    return {A, concerned.empty() ? -1 : concerned.back()};
}

std::unordered_set<int> LegoBlockGraph::k_neighbourhood(int root_id, int k) const {
    std::unordered_set<int> S;
    if (!is_alive(root_id)) return S;

    std::vector<int> dist(bricks_.size(), -1);
    std::deque<int> q;
    dist[root_id] = 0;
    q.push_back(root_id);

    while (!q.empty()) {
        int u = q.front(); q.pop_front();
        if (dist[u] >= 0 && dist[u] < k) {
            for (int v : get_neighbours_all(u)) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    S.insert(v);
                    q.push_back(v);
                }
            }
        }
    }
    return S;
}

std::vector<int> LegoBlockGraph::split_to_unit(int id, std::vector<std::array<int,5>>& unit_specs) const {
    std::vector<int> dummy; // placeholder, not used
    if (!is_alive(id)) return dummy;
    const auto& b = bricks_[id];
    for (int dx = 0; dx < b.sx; ++dx) {
        for (int dy = 0; dy < b.sy; ++dy) {
            unit_specs.push_back({b.x + dx, b.y + dy, b.z, 1, 1});
        }
    }
    return dummy;
}

LegoBlockGraph LegoBlockGraph::clone() const {
    LegoBlockGraph cpy;
    cpy.K_N = K_N;
    cpy.F_MAX = F_MAX;
    cpy.bricks_ = bricks_;
    cpy.graph_ = graph_;
    cpy.reverse_graph_ = reverse_graph_;
    cpy.horizontal_neigh_ = horizontal_neigh_;
    cpy.levels_ = levels_;
    cpy.visited_ = visited_;
    cpy.rng_ = rng_; // copy RNG state
    cpy.voxel_grid_ = voxel_grid_; // 必须克隆地图
    cpy.max_X_ = max_X_;
    cpy.max_Y_ = max_Y_;
    cpy.max_Z_ = max_Z_;
    return cpy;
}

LegoBlockGraph LegoBlockGraph::shuffle(int w_id) {
    LegoBlockGraph repl = this->clone();
    if (!repl.is_alive(w_id)) return repl;

    auto neigh = repl.k_neighbourhood(w_id, repl.K_N);
    neigh.insert(w_id);

    std::vector<std::array<int,5>> replacements;
    for (int id : neigh) {
        if (!repl.is_alive(id)) continue;
        repl.split_to_unit(id, replacements);
        repl.remove_block(id);
    }

    for (auto &u : replacements) {
        repl.add_block(u[0], u[1], u[2], u[3], u[4]);
    }
    //repl.merge_to_maximal();
    return repl;
}


int LegoBlockGraph::generate_single_component_analysis(int max_iter) {
    auto [A, w_id] = component_analysis();
    int f = 0;

    while (A > 1 && f < max_iter) {
        // 加快 K 的增长速度，确保能覆盖到周围的支撑点
        K_N = (f / 10) + 1; 
        //std::cout<<"f"<<f<<" ";
        if (w_id < 0 || !is_alive(w_id)) {
            auto tmp = component_analysis();
            A = tmp.first; w_id = tmp.second;
            if (A <= 1) break;
        }

        LegoBlockGraph trial = this->clone();
        trial = trial.shuffle(w_id); 
        
        // 关键：交替变换合并方向。偶数次优先X合并，奇数次优先Y合并
        // 这能极大增加跳出“碎片化陷阱”的概率
        trial.merge_to_maximal(f % 2); 

        auto [A2, w2] = trial.component_analysis();

        if (A2 < A) {
            *this = std::move(trial);
            A = A2; w_id = w2; f = 0;
        } else {
            ++f;
        }
    }
    return A;
}


SolveResult LegoBlockGraph::export_result() const {
    SolveResult res;
    // components (fresh compute on a copy to keep const)
    {
        LegoBlockGraph tmp = this->clone();
        auto p = tmp.component_analysis();
        res.components = p.first;
    }

    for (const auto& b : bricks_) {
        if (!b.alive) continue;
        res.bricks.push_back({b.x, b.y, b.z, b.sx, b.sy});
    }

    std::sort(res.bricks.begin(), res.bricks.end(),
        [](const auto& a, const auto& b){
            if (a[0] != b[0]) return a[0] < b[0];
            if (a[1] != b[1]) return a[1] < b[1];
            return a[2] < b[2];
        });

    return res;
}
