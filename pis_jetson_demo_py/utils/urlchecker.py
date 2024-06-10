from enum import Enum


class URLType(Enum):
    UNKNOWN = -1
    FILE = 0
    X11 = 1
    LIVE = 2
    VIDEO_FOLDER = 3
    NONE = 100


# X11_IDENTIFIER_REGEX = re.compile(r"^[a-zA-Z0-9\.\-_]*:[0-9]+(\.[0-9]+)?$")
# FILE_IDENTIFIER_REGEX = re.compile(r"^((file:\/\/)|\/).*$")
# DEVICE_ID_REGEX = re.compile(r"^[0-9]+$")


def get_url_type(type_indicator: str) -> URLType:
    if type_indicator == "file":
        return URLType.FILE
    if type_indicator == "x11":
        return URLType.X11
    if type_indicator in ["rtsp", "live", "camera", "camera_ov2311"]:
        return URLType.LIVE
    if type_indicator == "video_folder":
        return URLType.VIDEO_FOLDER
    return URLType.UNKNOWN


# def check_url(url: str) -> Tuple[URLType, str]:
#     url = url.strip()
#     if re.match(FILE_IDENTIFIER_REGEX, url):
#         return URLType.FILE, url
#     if re.match(X11_IDENTIFIER_REGEX, url):
#         return URLType.X11, url
#     if re.match(DEVICE_ID_REGEX, url):
#         return URLType.LIVE, url
#     return URLType.FILE, url
