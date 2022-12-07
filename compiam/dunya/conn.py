# Copyright 2013, 2014 Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Dunya and has been ported from pycompmusic
# (https://github.com/MTG/pycompmusic), the official Python API
#
# Dunya is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free Software
# Foundation (FSF), either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see http://www.gnu.org/licenses/

import logging
import urllib.parse as urllibparse

import requests
import requests.adapters

logger = logging.getLogger("dunya")

from compiam.exceptions import HTTPError, ConnectionError

HOSTNAME = "https://dunya.compmusic.upf.edu"
TOKEN = None
session = requests.Session()
session.mount("http://", requests.adapters.HTTPAdapter(max_retries=5))
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=5))


def set_hostname(hostname):
    """Change the hostname of the dunya API endpoint.

    :param hostname: The new dunya hostname to set. If you want to access over http or a different port,
         include them in the hostname, e.g. `http://localhost:8000`.
    :returns: None.
    """
    global HOSTNAME
    HOSTNAME = hostname


def set_token(token):
    """Get an access token. You must call this before you can make.

    :param token: your access token.
    :returns: None.
    """
    global TOKEN
    TOKEN = token


def _get_paged_json(path, **kwargs):
    extra_headers = None
    if "extra_headers" in kwargs:
        extra_headers = kwargs.get("extra_headers")
        del kwargs["extra_headers"]
    nxt = _make_url(path, **kwargs)
    logger.debug("initial paged to %s", nxt)
    ret = []
    while nxt:
        res = _dunya_url_query(nxt, extra_headers=extra_headers)
        res = res.json()
        ret.extend(res.get("results", []))
        nxt = res.get("next")
    return ret


def _dunya_url_query(url, extra_headers=None):
    logger.debug("query to '%s'" % url)
    if not TOKEN:
        raise ConnectionError("You need to authenticate with `set_token`")

    headers = {"Authorization": "Token %s" % TOKEN}
    if extra_headers:
        headers.update(extra_headers)

    g = session.get(url, headers=headers)
    try:
        g.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise HTTPError(e)
    return g


def _dunya_post(url, data=None, files=None):
    data = data or {}
    files = files or {}
    logger.debug("post to '%s'" % url)
    if not TOKEN:
        raise ConnectionError("You need to authenticate with `set_token`")
    headers = {"Authorization": "Token %s" % TOKEN}
    p = requests.post(url, headers=headers, data=data, files=files)
    try:
        p.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise HTTPError(e)
    return p


def _make_url(path, **kwargs):
    if "://" in HOSTNAME:
        protocol, hostname = HOSTNAME.split("://")
    else:
        protocol = "http"
        hostname = HOSTNAME

    if not kwargs:
        kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            kwargs[key] = value.encode("utf8")
    url = urllibparse.urlunparse(
        (protocol, hostname, "%s" % path, "", urllibparse.urlencode(kwargs), "")
    )
    return url


def _dunya_query_json(path, **kwargs):
    """Make a query to dunya and expect the results to be JSON."""
    g = _dunya_url_query(_make_url(path, **kwargs))
    return g.json() if g else None


def _dunya_query_file(path, **kwargs):
    """Make a query to dunya and return the raw result."""
    g = _dunya_url_query(_make_url(path, **kwargs))
    if g:
        cl = g.headers.get("content-length")
        content = g.content
        if cl and int(cl) != len(content):
            logger.warning(
                "Indicated content length is not the same as returned content. Some data may be missing"
            )
        return content
    else:
        return


def _file_for_document(recording_id, thetype, subtype=None, part=None, version=None):
    """Get the most recent derived file given a filetype.

    :param recording_id: Musicbrainz recording ID.
    :param thetype: the computed filetype.
    :param subtype: a subtype if the module has one.
    :param part: the file part if the module has one.
    :param version: a specific version, otherwise the most recent one will be used.
    :returns: The contents of the most recent version of the derived file.
    """
    path = "document/by-id/%s/%s" % (recording_id, thetype)
    args = {}
    if subtype:
        args["subtype"] = subtype
    if part:
        args["part"] = part
    if version:
        args["v"] = version
    return _dunya_query_file(path, **args)


def get_mp3(recording_id):
    """Get a mp3 from a specific mbid.

    :param recording_id: Musicbrainz recording ID.
    """
    return _file_for_document(recording_id, "mp3")
