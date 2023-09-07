from pathlib import Path
import shlex
from subprocess import Popen, PIPE

def animate_pngs(label:str, image_dir, out_path:Path, framerate:int=15):
    """
    Given a directory with png files following a {label}_{order_str}.png
    format with unique identifying string labels, and globbable order strings,
    use ffmpeg to generate an animation (ie mp4, gif) from all matching
    PNGs.

    :@param label: String label matching the first underscore-separated field
        of all png files to incorporate into the animation
    :@param image_dir: Directory containing images to animate
    :@param out_path: Full path to a video file output.
    :@param framerate: Animation framerate

    :@return: ffmpeg subprocess communication object
    """
    #seq_str = " -i ".join(video_seqs[l])
    #out_path = video_dir.joinpath(l).as_posix()+".gif"
    #print(seq_str)
    ffmpeg_cmd = f"ffmpeg -hide_banner -loglevel error -f image2 "+ \
            f"-r {framerate} -pattern_type glob -i " + \
            f"'{image_dir.as_posix()}/{l}_*.png' -crf 22 {out_path}"
    proc = Popen(shlex.split(ffmpeg_cmd), stdout=PIPE)
    return proc.communicate()

def text_on_mp4(in_path:Path, out_path, text:str, color="white",
                size=24, offset:tuple=(0,0), box=1, wbox=2, obox=.5,
                font="/usr/share/fonts/TTF/FiraCodeNerdFontMono-Regular.ttf"):
    """
    Write static text on an MP4 video with an optional background box.

    :@param offset:2-tuple of values in [0,1] representing (respectively) the
        vertical and horizontal percentage offset of the top left corner of the
        text in terms of the dimensions of the full image.
    """
    filter_dict = {
            "drawtext=fontfile":font,
            "text":text,
            "fontcolor":color,
            "fontsize":size,
            "box":box,
            "boxcolor":f"{color}@{obox}",
            "boxborderw":wbox,
            "x":f"{offset[1]}*w",
            "y":f"{offset[0]}*h",
            }
    filter_str = ":".join("=".join(map(str,t)) for t in filter_dict.items())
    ffmpeg_cmd = f"ffmpeg -i {in_path.as_posix()} -vf \"{filter_str}\" " + \
            f"-qscale 0 -codec:a copy {out_path.as_posix()}"
    proc = Popen(shlex.split(ffmpeg_cmd), stdout=PIPE)
    return proc.communicate()

if __name__=="__main__":
    image_dir = Path("figures/video_factory")
    video_dir = Path(f"figures/videos")
    video_ext = "mp4"
    framerate = 8

    # Filenames expected to have 2 underscore-separated fields like:
    # {label}_{YYYYmmdd-HHMM}.png
    fields = [p for p in map(lambda p:(*p.stem.split("_"),p.as_posix()),
                             image_dir.iterdir())]
    # Get a list of unique labels
    labels = list(set(f[0] for f in fields))
    # Get time-ordered list of files for each label
    video_seqs = {
            l:[f[2] for f in sorted(fields, key=lambda f:f[1]) if l in f[2]]
            for l in labels
            }
    for l in labels:
        video_path = video_dir.joinpath(l+"."+video_ext)
        animate_pngs(l, image_dir, video_path, framerate=framerate)
        text_on_mp4(
                in_path=video_path,
                out_path=video_path.parent.joinpath("text_"+video_path.name),
                text=l,
                offset=(0,0)
                )
