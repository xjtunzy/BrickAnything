"""
2-20
"""

import random
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import networkx as nx

class LegoBrick:
    __slots__ = ('x','y','z','sx','sy','w')
    def __init__(self,x:int,y:int,z:int,sx:int,sy:int):
        self.x,  self.y,  self.z = x ,  y,  z
        self.sx, self.sy         = sx, sy
        self.w = 0
    @property
    def xspan(self): return (self.x, self.x + self.sx)
    @property
    def yspan(self): return (self.y, self.y + self.sy)
    @staticmethod
    def has_valid_size(sx:int, sy:int)->bool:
        a,b = sorted((sx, sy))
        #if a==1 and b in (1,2,3,4,6,8): return True
        if a==1 and b in (1,2,4,6,8): return True
        #if a==2 and b in (2,3,4,6,8):   return True
        if a==2 and b in (2,4,6):   return True
        return False
    
class LegoBrickUtils:
    @staticmethod
    def _overlap_1d(a:Tuple[int,int], b:Tuple[int,int])->bool:
        # 半开区间 [l, r)
        return not (a[1] <= b[0] or b[1] <= a[0])
    @staticmethod
    def are_connected(a:LegoBrick, b:LegoBrick)->bool:
        """ 垂直方向有重叠 """
        if abs(a.z - b.z) != 1: return False
        return (LegoBrickUtils._overlap_1d(a.xspan, b.xspan) and
                LegoBrickUtils._overlap_1d(a.yspan, b.yspan))
    @staticmethod
    def are_adjacent(a:LegoBrick, b:LegoBrick)->bool:
        """ 同层相邻 """
        if a.z != b.z: return False
        # x相邻 + y重叠
        x_touch = (a.xspan[1] == b.xspan[0]) or (b.xspan[1] == a.xspan[0])
        y_ovl   = LegoBrickUtils._overlap_1d(a.yspan, b.yspan)
        # y相邻 + x重叠
        y_touch = (a.yspan[1] == b.yspan[0]) or (b.yspan[1] == a.yspan[0])
        x_ovl   = LegoBrickUtils._overlap_1d(a.xspan, b.xspan)
        return (x_touch and y_ovl) or (y_touch and x_ovl)
    @staticmethod
    def can_merge(a:LegoBrick,b:LegoBrick)->bool:
        """判定两块是否能在同层合并成一个更大的brick"""
        if a.z != b.z: return False
        if a.yspan == b.yspan:
            # try x axis merge
            xs = sorted([a.x, b.x])
            ys = a.yspan
            sx = (a.sx + b.sx) if xs[0] + a.sx == xs[1] or  xs[0] + b.sx == xs[1] else None
            if sx is not None:
                if LegoBrick.has_valid_size(sx,a.sy):
                    return True 
        if a.xspan == b.xspan:
            ys = sorted([a.y, b.y])
            xs = a.xspan
            sy = (a.sy + b.sy) if ys[0] + a.sy == ys[1] or ys[0] + b.sy == ys[1] else None
            if sy is not None:
                if LegoBrick.has_valid_size(a.sx, sy):
                    return True
        return False
    @staticmethod
    def merge(a:LegoBrick, b:LegoBrick)->LegoBrick:
        """ 合并出新的Brick """
        assert a.z == b.z
        # X向拼接
        if a.yspan == b.yspan and (a.xspan[1] == b.xspan[0] or b.xspan[1] == a.xspan[0]):
            x = min(a.x, b.x)
            y = a.y
            sx = a.sx + b.sx
            sy = a.sy
            return LegoBrick(x, y, a.z, sx, sy)
        # Y向拼接
        if a.xspan == b.xspan and (a.yspan[1] == b.yspan[0] or b.yspan[1] == a.yspan[0]):
            x = a.x
            y = min(a.y, b.y)
            sx = a.sx
            sy = a.sy + b.sy
            return LegoBrick(x, y, a.z, sx, sy)
        raise RuntimeError("merge called on non-mergeable pair")


    


class LegoBlockGraph():
    
    def __init__(self):
        self.K_N = 2     # shuffle时的邻域深度
        self.F_MAX = 200 # 最大失败次数
        self.blocks: List[LegoBrick] = []
        self.graph : Dict[LegoBrick,List[LegoBrick]] = defaultdict(list)
        self.reverse_graph: Dict[LegoBrick,List[LegoBrick]] = defaultdict(list)
        self.horizontal_neighbours: Dict[LegoBrick,List[LegoBrick]] = defaultdict(list)
        self.levels: Dict[int,List[LegoBrick]] = defaultdict(list)
        self.visited: Dict[LegoBrick,int] = {}
    
    @staticmethod
    def from_numpy(occ: np.ndarray) -> "LegoBlockGraph":
        g = LegoBlockGraph()
        X,Y,Z = occ.shape
        for z in range(Z):
            for y in range(Y):
                xs = np.where(occ[:,y,z]>0)[0]
                for x in xs:
                    g.add_block(LegoBrick(x, y, z, 1, 1))
        #print(len(g.graph))
        return g
    
    def get_neighbours_all(self, block:LegoBrick)->List[LegoBrick]:
        return list(self.graph[block]) + list(self.reverse_graph[block]) + list(self.horizontal_neighbours[block])

    def _filter_connected(self, blocks:List[LegoBrick], ref:LegoBrick)->List[LegoBrick]:
        return [b for b in blocks if LegoBrickUtils.are_connected(b, ref)]

    def _get_horizontal_neighbours_from_all(self, ref:LegoBrick)->List[LegoBrick]:
        return [b for b in self.blocks if b is not ref and LegoBrickUtils.are_adjacent(b, ref)]

    def add_block(self, block:LegoBrick):
        self.blocks.append(block)
        self.levels[block.z].append(block)
        # children on z+1
        self.graph[block] = self._filter_connected(self.levels.get(block.z+1, []), block)
        for nb in self.graph[block]:
            self.reverse_graph[nb].append(block)
        # parents on z-1
        self.reverse_graph[block] = self._filter_connected(self.levels.get(block.z-1, []), block)
        for nb in self.reverse_graph[block]:
            self.graph[nb].append(block)
        # horizontal neighbours
        self.horizontal_neighbours[block] = self._get_horizontal_neighbours_from_all(block)
        for nb in self.horizontal_neighbours[block]:
            self.horizontal_neighbours[nb].append(block)

    def remove_block(self, block:LegoBrick):
        if block in self.blocks: self.blocks.remove(block)
        if block in self.levels.get(block.z, []): self.levels[block.z].remove(block)
        for ch in self.graph.get(block, []): 
            if block in self.reverse_graph[ch]: self.reverse_graph[ch].remove(block)
        for pa in self.reverse_graph.get(block, []):
            if block in self.graph[pa]: self.graph[pa].remove(block)
        for nb in self.horizontal_neighbours.get(block, []):
            if block in self.horizontal_neighbours[nb]: self.horizontal_neighbours[nb].remove(block)
        self.graph.pop(block, None)
        self.reverse_graph.pop(block, None)
        self.horizontal_neighbours.pop(block, None)

    # ---------- 合并 ----------
    def _generate_mergables(self):
        pairs = []
        seen = set()
        for a in self.blocks:
            for b in self.horizontal_neighbours[a]:
                if (id(b), id(a)) in seen: 
                    continue
                if LegoBrickUtils.can_merge(a,b):
                    pairs.append((a,b))
                    seen.add((id(a), id(b)))
        return pairs

    def merge_to_maximal(self):
        mergables = self._generate_mergables()
        while mergables:
            a,b = random.choice(mergables)
            newb = LegoBrickUtils.merge(a,b)
            # 清理与添加
            # 删除所有与a/b相关的候选
            mergables = [(x,y) for (x,y) in mergables if x not in (a,b) and y not in (a,b)]
            self.remove_block(a)
            self.remove_block(b)
            self.add_block(newb)
            # 给新块添加新的候选
            for nb in self.horizontal_neighbours[newb]:
                if LegoBrickUtils.can_merge(newb, nb):
                    mergables.append((newb, nb))

    # ---------- 分析 ----------
    def component_analysis(self):
        """ 仅通过上下连接（graph & reverse_graph）做连通分量标号；
            返回 (组件数, 一个“候选问题块”指针)。"""
        self.visited = {b:-1 for b in self.blocks}
        #print(self.visited)
        A = 0
        for bi in self.blocks:
            if self.visited[bi] != -1: 
                continue
            # BFS 在垂直连接上
            Q = deque([bi])
            self.visited[bi] = A
            while Q:
                u = Q.popleft()
                for v in self.graph[u] + self.reverse_graph[u]:
                    if self.visited[v] == -1:
                        self.visited[v] = A
                        Q.append(v)
            A += 1

        # 为每个块计算“它与其一阶邻居覆盖了多少不同组件”
        concerned = []
        total_w = 0
        for b in self.blocks:
            comps = { self.visited[b] }
            for nb in self.get_neighbours_all(b):
                comps.add(self.visited[nb])
            b.w = len(comps) - 1
            total_w += b.w
            if b.w > 0:
                concerned.append(b)

        if total_w == 0:
            # 已经是单连通，随便返回一个块即可
            return (A, self.blocks[0] if self.blocks else None)

        # 按 w 加权随机选一个“问题块”
        pick = random.randrange(total_w)
        acc = 0
        for b in concerned:
            acc += b.w
            if acc > pick:
                return (A, b)
        return (A, concerned[-1] if concerned else (self.blocks[0] if self.blocks else None))

    def _k_neighbourhood(self, root:LegoBrick, k:int)->Set[LegoBrick]:
        dist = {b:-1 for b in self.blocks}
        dist[root] = 0
        S: Set[LegoBrick] = set()
        Q = deque([root])
        while Q:
            u = Q.popleft()
            if 0 <= dist[u] < k:
                for v in self.get_neighbours_all(u):
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        S.add(v); Q.append(v)
        return S

    def _split_to_unit(self, block:LegoBrick)->List[LegoBrick]:
        res=[]
        for dx in range(block.sx):
            for dy in range(block.sy):
                res.append(LegoBrick(block.x+dx, block.y+dy, block.z, 1,1))
        return res

    def shuffle(self, w:LegoBrick)->"LegoBlockGraph":
        """ 复制图，取 w 的 K 邻域 + w 自身 -> 拆回 1×1 -> 再合并到极大，返回新图 """
        repl = LegoBlockGraph()
        # 深拷贝
        for b in self.blocks:
            b2 = LegoBrick(b.x,b.y,b.z,b.sx,b.sy)
            repl.add_block(b2)
        # 找到复制后的 w'
        #（简化：按坐标匹配）
        candidate = None
        for b in repl.blocks:
            if (b.x,b.y,b.z,b.sx,b.sy)==(w.x,w.y,w.z,w.sx,w.sy):
                candidate = b; break

        neighbourhood = repl._k_neighbourhood(candidate, self.K_N)
        neighbourhood.add(candidate)

        # 移除并拆回 1x1
        replacements=[]
        for nb in list(neighbourhood):
            repl.remove_block(nb)
            replacements.extend(repl._split_to_unit(nb))
        for r in replacements:
            repl.add_block(r)
        repl.merge_to_maximal()
        return repl

    def generate_single_component_analysis(self, max_iter:int=200):
        A, w = self.component_analysis()
        f = 0
        while A > 1 and f < max_iter:
            self.K_N = f//10+1
            trial = self.shuffle(w)
            A2, w2 = trial.component_analysis()
            if A2 < A:
                # 接受新解
                self.__dict__ = trial.__dict__
                A, w = A2, w2
                f = 0
                #print(f"new structure  components:{A}")
            else:
                f += 1
                #print(f"num: {f}")
        return A
