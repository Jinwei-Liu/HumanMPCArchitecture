import math

inner_radius = 0.5      # 内圈半径 0.5 m
thickness = 0.2       
outer_radius = inner_radius + thickness
segments = 64          

with open("gate.obj", "w") as f:
    for r in (inner_radius, outer_radius):
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            x = r * math.cos(theta)
            y = 0.0
            z = r * math.sin(theta)
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
    for _ in range(2 * segments):
        f.write("vn 0 1 0\n")
    for i in range(segments):
        i_inner1 = i + 1
        i_inner2 = (i + 1) % segments + 1
        i_outer1 = segments + i + 1
        i_outer2 = segments + ((i + 1) % segments) + 1
        # f v1//vn1 v2//vn2 v3//vn3 v4//vn4
        f.write(f"f {i_inner1}//{i_inner1} {i_inner2}//{i_inner2} "
                f"{i_outer2}//{i_outer2} {i_outer1}//{i_outer1}\n")
