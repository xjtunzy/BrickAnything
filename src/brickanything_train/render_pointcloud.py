import os
import numpy as np
import mitsuba as mi
import tempfile

# ---------- 复用你原来的 XML 模板 ----------
xml_head = """
<scene version="3.0.0">
    <integrator type="path"/>

    <sensor type="perspective">
        <transform name="to_world">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>

        <sampler type="independent">
            <integer name="sample_count" value="256"/>
        </sampler>

        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>

    <bsdf type="roughplastic" id="surfaceMaterial">
        <float name="alpha" value="0.05"/>
        <float name="int_ior" value="1.46"/>
        <float name="ext_ior" value="1.0"/>
        <rgb name="diffuse_reflectance" value="1,1,1"/>
    </bsdf>
"""

xml_ball_segment = """
    <shape type="sphere">
        <float name="radius" value="0.025"/>
        <transform name="to_world">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="to_world">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>

    <shape type="rectangle">
        <transform name="to_world">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>

</scene>
"""

def _standardize_bbox_xyz(pcl_xyz: np.ndarray, points_per_object: int = 2048, seed: int | None = None):
    assert pcl_xyz.ndim == 2 and pcl_xyz.shape[1] == 3, "pcl must be (N,3)"
    rng = np.random.default_rng(seed)

    N = pcl_xyz.shape[0]
    K = min(points_per_object, N)
    idx = rng.choice(N, size=K, replace=False)
    pcl = pcl_xyz[idx].astype(np.float32)

    mins = pcl.min(axis=0)
    maxs = pcl.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = np.max(maxs - mins)
    scale = max(float(scale), 1e-8)

    pcl = (pcl - center) / scale
    return pcl

def render_pointcloud_xyz_to_png(
    pcl_xyz: np.ndarray,
    png_out_path: str,
    points_per_object: int = 2048,
    spp: int = 256,
    variant: str = "cuda_ad_rgb",
    color=(0.239, 0.439, 0.847),
):
    """
    输入:
      pcl_xyz: (N,3) numpy array
      png_out_path: 保存 png 的路径
    输出:
      png_out_path (文件已写入)
    """
    # 1) 采样 + 标准化
    pcl = _standardize_bbox_xyz(pcl_xyz, points_per_object)

    # 2) 复用你原来的坐标变换
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125  # match sphere radius

    # 3) 生成 XML 字符串（不需要你手动提供 xml_out_path）
    xml_segments = [xml_head]
    for i in range(pcl.shape[0]):
        xml_segments.append(xml_ball_segment.format(
            pcl[i, 0], pcl[i, 1], pcl[i, 2],
            color[0], color[1], color[2]
        ))
    xml_segments.append(xml_tail)
    xml_content = "".join(xml_segments)

    # 4) 写临时 XML 文件 -> mitsuba load_file 渲染
    os.makedirs(os.path.dirname(png_out_path) or ".", exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
        xml_path = tf.name
        tf.write(xml_content.encode("utf-8"))

    mi.set_variant(variant)
    scene = mi.load_file(xml_path)
    image = mi.render(scene, spp=spp)

    mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8).write(png_out_path)

    # 5) 清理临时 XML
    try:
        os.remove(xml_path)
    except OSError:
        pass

    return png_out_path

