from ipywidgets import HTML
from pathlib import Path
import mimetypes
import json
import base64
import jinja2
import os
import html
import pathlib

WORKDIR = os.path.dirname(pathlib.Path(__file__).parent.resolve())
TOOLDIR = os.path.join(WORKDIR, "waveform_player", "")
WPDIR = os.path.join(TOOLDIR, "waveform-playlist", "")


def audio_file_to_base64(filename):
    mimetype = mimetypes.guess_type(filename)
    return (
        "data:"
        + mimetype[0]
        + ";base64,"
        + base64.b64encode(Path(filename).read_bytes()).decode("ascii")
    )


def json_track_list(titles, files, gains, mutes, solos):
    assert (
        len(titles) == len(files)
        and (gains == None or len(gains) == len(titles))
        and (mutes == None or len(mutes) == len(titles))
        and (solos == None or len(solos) == len(titles))
    )
    res = []
    for i in range(len(titles)):
        entry = {}
        entry["src"] = audio_file_to_base64(files[i])
        entry["name"] = titles[i]
        if gains is not None:
            entry["gain"] = gains[i]
        if mutes is not None:
            entry["muted"] = mutes[i]
        if solos is not None:
            entry["soloed"] = solos[i]
        res.append(entry)
    return json.dumps(res)


def local_text(filename):
    if filename[0] == "/":
        filename = filename[1:]  # this is a bit ghetto
    full_path = os.path.join(TOOLDIR, filename)
    with open(full_path, "r") as f:
        return f.read()


def make_playlist_iframe(
    titles,
    files,
    gains=None,
    mutes=None,
    solos=None,
    annotations=None,
    template_name="multi-channel.html",
):
    templateLoader = jinja2.FileSystemLoader(searchpath=WPDIR)
    templateEnv = jinja2.Environment(loader=templateLoader)
    print(WPDIR)
    print(template_name)
    template = templateEnv.get_template(template_name)

    return template.render(
        {
            "local_text": local_text,
            "play_list_json": lambda: json_track_list(
                titles, files, gains, mutes, solos
            ),
            "annotations_data": lambda: json.dumps(annotations),
        }
    )


class Player(HTML):
    def __init__(
        self,
        titles,
        files,
        annotations=None,
        height=None,
        width=1000,
        gains=None,
        mutes=None,
        solos=None,
    ):
        if isinstance(titles, str):
            titles = [titles]
        if isinstance(files, str):
            files = [files]
        assert len(titles) == len(
            files
        ), "Requires titles and files to be equal in length"
        if not height:
            # 180 headers and buttons, about 150 for each audio
            height = 180 + 150 * len(titles)
            if annotations:
                height += 200
        template_name = "annotations.html" if annotations else "multi-channel.html"
        super().__init__(
            '<iframe srcdoc="'
            + html.escape(
                make_playlist_iframe(
                    titles,
                    files,
                    gains,
                    mutes,
                    solos,
                    annotations=annotations,
                    template_name=template_name,
                )
            )
            + f'" height={height} width={width}></iframe>'
        )
