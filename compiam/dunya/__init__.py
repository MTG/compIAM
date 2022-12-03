# Copyright 2013, 2014 Music Technology Group - Universitat Pompeu Fabra
#
# Several functions in this file are part of Dunya and have been ported
# from pycompmusic (https://github.com/MTG/pycompmusic), the official API
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

import os
import errno

from .conn import (
    _get_paged_json,
    _dunya_query_json,
    _get_paged_json,
    get_mp3,
    _file_for_document,
    set_token,
)
from compiam.io import write_csv, write_json, write_scalar_txt


class Corpora:
    """Dunya corpora class with access functions"""

    def __init__(self, tradition, cc, token):
        """Dunya corpora class init method.

        :param tradition: the name of the tradition.
        :param cc: bool flag to indicate if the Creative Commons version of the corpora is chosen.
        :param token: Dunya personal token to access te database.
        """
        # Load and set token
        self.token = token
        set_token(self.token)

        self.tradition = tradition
        self.collection = (
            "dunya-" + self.tradition + "-cc" if cc else "dunya-" + self.tradition
        )

    def get_collection(self):
        """Get the documents (recordings) in a collection.

        :param slug: the name of the collection.
        """
        return _dunya_query_json("document/" + self.collection)["documents"]

    def list_recordings(self, recording_detail=False):
        """List the recordings in the database. This function will automatically page through API results.

        :param recording_detail: if True, return full details for each recording like :func:`get_recording`.
        :returns: A list of dictionaries containing recording information:
            ``{"mbid": MusicBrainz recording ID, "title": Title of the recording}``
            For additional information about each recording use :func:`get_recording`.
        """
        args = {}
        if recording_detail:
            args["detail"] = "1"
        return _get_paged_json("api/" + self.tradition + "/recording", **args)

    def get_recording(self, rmbid):
        """Get specific information about a recording.

        :param rmbid: A recording MBID.
        :returns: mbid, title, artists, raga, tala, work.
            ``artists`` includes performance relationships attached to the recording, the release, and the release artists.
        """
        return _dunya_query_json("api/" + self.tradition + "/recording/%s" % rmbid)

    def list_artists(self):
        """List the artists in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing artist information:
            ``{"mbid": MusicBrainz artist id, "name": Name of the artist}``.
            For additional information about each artist use :func:`get_artist`.
        """
        return _get_paged_json("api/" + self.tradition + "/artist")

    def get_artist(self, ambid):
        """Get specific information about an artist.

        :param ambid: An artist MBID.
        :returns: mbid, name, concerts, instruments, recordings.
            ``concerts``, ``instruments`` and ``recordings`` include
            information from recording- and release-level
            relationships, as well as release artists.
        """
        return _dunya_query_json("api/" + self.tradition + "/artist/%s" % (ambid))

    def list_concerts(self):
        """List the concerts in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing concert information:
            ``{"mbid": MusicBrainz Release ID, "title": title of the concert}``
            For additional information about each concert use :func:`get_concert`.
        """
        return _get_paged_json("api/" + self.tradition + "/concert")

    def get_concert(self, cmbid):
        """Get specific information about a concert.

        :param cmbid: A concert mbid.
        :returns: mbid, title, artists, tracks.
            ``artists`` includes performance relationships attached
            to the recordings, the release, and the release artists.
        """
        return _dunya_query_json("api/" + self.tradition + "/concert/%s" % cmbid)

    def list_works(self):
        """List the works in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing work information:
            ``{"mbid": MusicBrainz work ID, "name": work name}``
            For additional information about each work use :func:`get_work`.
        """
        return _get_paged_json("api/" + self.tradition + "/work")

    def get_work(self, wmbid):
        """Get specific information about a work.

        :param wmbid: A work mbid.
        :returns: mbid, title, composers, ragas, talas, recordings.
        """
        return _dunya_query_json("api/" + self.tradition + "/work/%s" % (wmbid))

    def list_ragas(self):
        """List the ragas in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing raga information:
            ``{"uuid": raga UUID, "name": name of the raga}``
            For additional information about each raga use :func:`get_raga`.
        """
        return _get_paged_json("api/" + self.tradition + "/raga")

    def get_raga(self, raga_id):
        """Get specific information about a raga.

        :param raga_id: A raga id or uuid.
        :returns: uuid, name, artists, works, composers.
            ``artists`` includes artists with recording- and release-
            level relationships to a recording with this raga.
        """
        return _dunya_query_json("api/" + self.tradition + "/raga/%s" % str(raga_id))

    def list_talas(self):
        """List the talas in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing tala information:
            ``{"uuid": tala UUID, "name": name of the tala}``
            For additional information about each tala use :func:`get_tala`.
        """
        return _get_paged_json("api/" + self.tradition + "/tala")

    def get_tala(self, tala_id):
        """Get specific information about a tala.

        :param tala_id: A tala id or uuid.
        :returns: uuid, name, artists, works, composers.
            ``artists`` includes artists with recording- and release-
            level relationships to a recording with this raga.
        """
        return _dunya_query_json("api/" + self.tradition + "/tala/%s" % str(tala_id))

    def list_instruments(self):
        """List the instruments in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing instrument information:
            ``{"id": instrument id, "name": Name of the instrument}``
            For additional information about each instrument use :func:`get_instrument`.
        """
        return _get_paged_json("api/" + self.tradition + "/instrument")

    def get_instrument(self, instrument_id):
        """Get specific information about an instrument.

        :param instrument_id: An instrument id
        :returns: id, name, artists.
            ``artists`` includes artists with recording- and release-
            level performance relationships of this instrument.
        """
        return _dunya_query_json("api/" + self.tradition + "/instrument/%s" % str(instrument_id))

    @staticmethod
    def list_available_types(recording_id):
        """Get the available source filetypes for a Musicbrainz recording.

        :param recording_id: Musicbrainz recording ID.
        :returns: a list of filetypes in the database for this recording.
        """
        document = _dunya_query_json("document/by-id/%s" % recording_id)
        return {
            x: list(document["derivedfiles"][x].keys())
            for x in list(document["derivedfiles"].keys())
        }

    @staticmethod
    def get_annotation(recording_id, thetype, subtype=None, part=None, version=None):
        """Alias function of _file_for_document in the Corpora class.

        :param recording_id: Musicbrainz recording ID.
        :param thetype: the computed filetype.
        :param subtype: a subtype if the module has one.
        :param part: the file part if the module has one.
        :param version: a specific version, otherwise the most recent one will be used.
        :returns: The contents of the most recent version of the derived file.
        """
        return _file_for_document(
            recording_id, thetype, subtype=subtype, part=part, version=version
        )

    @staticmethod
    def save_annotation(
        recording_id, thetype, location, subtype=None, part=None, version=None
    ):
        """A version of get_annotation that writes the parsed data into a file.

        :param recording_id: Musicbrainz recording ID.
        :param thetype: the computed filetype.
        :param subtype: a subtype if the module has one.
        :param part: the file part if the module has one.
        :param version: a specific version, otherwise the most recent one will be used.
        :returns: None (a file containing the parsed data is written).
        """
        data = _file_for_document(
            recording_id, thetype, subtype=subtype, part=part, version=version
        )
        if ("tonic" in subtype) or ("aksharaPeriod" in subtype):
            write_scalar_txt(data, location)
        elif "section" in subtype:
            write_json(data, location)
        elif "APcurve" in subtype:
            write_csv(data, location)
        elif ("pitch" in subtype) or ("aksharaTicks" in subtype):
            write_csv(data, location)
        else:
            raise ValueError(
                "No writing method available for data type: {} and {}", thetype, subtype
            )

    def download_mp3(self, recording_id, output_dir):
        """Download the mp3 of a document and save it to the specificed directory.

        :param recording_id: The MBID of the recording.
        :param output_dir: Where to save the mp3 to.
        :returns: name of the saved file.
        """
        if not os.path.exists(output_dir):
            raise Exception("Output directory %s doesn't exist; can't save" % output_dir)

        recording = self.get_recording(recording_id)
        if "concert" in list(recording.keys()):
            concert = self.get_concert(recording["concert"][0]["mbid"])
            title = recording["title"]
            artists = " and ".join([a["name"] for a in concert["concert_artists"]])
            name = "%s - %s.mp3" % (artists, title)
            name = name.replace("/", "-")
        else:
            name = recording_id + ".mp3"
        contents = get_mp3(recording_id)
        path = os.path.join(output_dir, name)
        open(path, "wb").write(contents)
        return name

    def download_concert(self, concert_id, output_dir):
        """Download the mp3s of all recordings in a concert and save them to the specificed directory.

        :param concert_id: The MBID of the concert.
        :param location: Where to save the mp3s to.
        """
        if not os.path.exists(output_dir):
            raise Exception("Output directory %s doesn't exist; can't save" % output_dir)

        concert = self.get_concert(concert_id)
        artists = " and ".join([a["name"] for a in concert["concert_artists"]])
        concertname = concert["title"]
        concertdir = "%s - %s" % (artists, concertname)
        concertdir = concertdir.replace("/", "-")
        concertdir = os.path.join(output_dir, concertdir)
        try:
            os.makedirs(concertdir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(concertdir):
                pass
            else:
                raise

        for r in concert["recordings"]:
            raga_id = r["mbid"]
            title = r["title"]
            disc = r["disc"]
            disctrack = r["disctrack"]
            contents = get_mp3(raga_id)
            name = "%s - %s - %s - %s.mp3" % (disc, disctrack, artists, title)
            path = os.path.join(concertdir, name)
            open(path, "wb").write(contents)
