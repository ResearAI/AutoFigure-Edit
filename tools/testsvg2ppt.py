from svg2ppt import convert_svg_to_ppt
with open("../outputs/demo/final.svg", "r") as f:
    svg_data = f.read()

convert_svg_to_ppt(svg_data, "result.pptx")
