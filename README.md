# Auto Image Rotator

I still shoot film, mostly half frame. The scans come back from the lab without any orientation info. A phone or digital camera knows how it was held, a film scanner doesn't. Half frame makes it worse: holding the camera normally gives a portrait picture, so a good part of every roll comes back sideways. Turning 72 photos per roll by hand gets old fast, so I wrote this to do it for me.

It looks at each JPEG with a small neural network, works out whether it's sideways or upside down, and sets the EXIF orientation tag. The image itself is never re-encoded.

```
$ ./run.sh --apply
[ 1/72] OK     photos/roll07_01.jpg: upright (99%)
[ 2/72] SET    photos/roll07_02.jpg: rotate 90 CCW (97%)
[ 3/72] SET    photos/roll07_03.jpg: rotate 180 (100%)
[ 4/72] UNSURE photos/roll07_04.jpg: rotate 90 CW? (61%) - left unchanged
...

1 image(s) need a manual look:
  photos/roll07_04.jpg: rotate 90 CW? (61%)
Turn them with e.g.:
  ./run.sh --rotate cw|ccw|180 <file>
```

## Usage

You need [Docker](https://docs.docker.com/engine/install/).

```bash
git clone https://github.com/juliankaroliny/auto-image-rotator.git
cd auto-image-rotator
cp ~/scans/roll07/*.jpg photos/

./run.sh           # dry run: shows what would change
./run.sh --apply   # set the tags
```

The first run builds the Docker image, which takes a few minutes. Files in `photos/` are changed in place, subfolders included.

Images the model isn't sure about are marked `UNSURE` and left alone. Turn those by hand:

```bash
./run.sh --rotate cw photos/img.jpg        # or ccw, 180
./run.sh --rotate upright photos/img.jpg   # undo
```

| Option | What it does |
|---|---|
| `--apply` | Write the tags (default is a dry run) |
| `--rotate TURN` | Set the orientation by hand: `cw`, `ccw`, `180` or `upright`. No detection |
| `--min-confidence X` | Below this, an image counts as unsure (default `0.8`) |
| `--force` | Also re-check images that already have an orientation set |

## How it works

The orientation comes from a ResNeXt-50 classifier from [check_orientation](https://github.com/ternaus/check_orientation), converted to ONNX during the Docker build. Each image is checked in all four rotations and the results are averaged, which makes the guess more reliable. The tool then writes the Orientation value into the EXIF header, adding the tag if the file doesn't have one yet, and leaves the image data as it is.

Setting the tag instead of really rotating is on purpose. Rotating a JPEG means re-encoding it, which costs quality. The "lossless" rotate in image viewers only works cleanly when the image size is a multiple of 16 px, otherwise it leaves a mirrored strip along one edge (with my half frame scans, that was always the case). The tag just tells viewers which way to show the image. Only the header changes (usually just two bytes), nothing is lost, and it can be undone anytime.

## Limitations

- The model gets most photos right, but not all. Indoor shots and close-ups are the usual suspects, so check the dry run.
- Only JPEG files are supported.
- A few programs ignore the tag and still show the image as scanned.

## License

MIT. The model weights come from [check_orientation](https://github.com/ternaus/check_orientation) (Apache-2.0). They are downloaded during the build and are not part of this repo.
