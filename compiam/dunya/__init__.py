import importlib

from compiam.io import (
    write_csv,
    write_json,
    write_scalar_txt,
)
from compmusic import dunya  # Using pycompmusic to access the Dunya database

from compiam.utils import get_logger

logger = get_logger(__name__)


class Corpora:
    """Dunya corpora class with access functions"""

    def __init__(self, tradition, token):
        """Dunya corpora class init method.

        :param tradition: the name of the tradition.
        :param token: Dunya personal token to access te database.
        """
        # Load and set token
        self.token = token
        dunya.set_token(self.token)

        # Load the tradition module. If the tradition is not available, pycompmusic raises ImportError.
        self.tradition = importlib.import_module(f"compmusic.dunya.{tradition}")

        logger.warning("""
            Note that a part of the collection is under restricted access.
            To access the full collection please request permission at https://dunya.compmusic.upf.edu/user/profile/
        """)

    def get_collection(self, verbose=False):
        """Get the documents (recordings) in a collection.

        :param verbose: Include additional information about each recording.
        :returns: dictionary with the recordings in the collection.
        """
        if verbose is False:
            logger.warning(
                "To parse the entire collection with all recording details, "
                + "please use the .get_collection(verbose=True) method. "
                + "Please note that it might take a few moments..."
            )
        return self.tradition.get_recordings(verbose)

    def get_recording(self, rmbid):
        """Get specific information about a recording.

        :param rmbid: A recording MBID.
        :returns: mbid, title, artists, raga, tala, work.
            ``artists`` includes performance relationships attached to the recording, the release, and the release artists.
        """
        return self.tradition.get_recording(rmbid)

    def get_artist(self, ambid):
        """Get specific information about an artist.

        :param ambid: An artist MBID.
        :returns: mbid, name, concerts, instruments, recordings.
            ``concerts``, ``instruments`` and ``recordings`` include
            information from recording- and release-level
            relationships, as well as release artists.
        """
        return self.tradition.get_artist(ambid)

    def list_concerts(self):
        """List the concerts in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing concert information:
            ``{"mbid": MusicBrainz Release ID, "title": title of the concert}``
            For additional information about each concert use :func:`get_concert`.
        """
        return self.tradition.get_concerts()

    def get_concert(self, cmbid):
        """Get specific information about a concert.

        :param cmbid: A concert mbid.
        :returns: mbid, title, artists, tracks.
            ``artists`` includes performance relationships attached
            to the recordings, the release, and the release artists.
        """
        return self.tradition.get_concert(cmbid)

    def list_works(self):
        """List the works in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing work information:
            ``{"mbid": MusicBrainz work ID, "name": work name}``
            For additional information about each work use :func:`get_work`.
        """
        return self.tradition.get_works()

    def get_work(self, wmbid):
        """Get specific information about a work.

        :param wmbid: A work mbid.
        :returns: mbid, title, composers, ragas, talas, recordings.
        """
        return self.tradition.get_work(wmbid)

    def list_ragas(self):
        """List the ragas in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing raga information:
            ``{"uuid": raga UUID, "name": name of the raga}``
            For additional information about each raga use :func:`get_raga`.
        """
        return self.tradition.get_raagas()

    def get_raga(self, raga_id):
        """Get specific information about a raga.

        :param raga_id: A raga id or uuid.
        :returns: uuid, name, artists, works, composers.
            ``artists`` includes artists with recording- and release-
            level relationships to a recording with this raga.
        """
        return self.tradition.get_raaga(raga_id)

    def list_talas(self):
        """List the talas in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing tala information:
            ``{"uuid": tala UUID, "name": name of the tala}``
            For additional information about each tala use :func:`get_tala`.
        """
        return self.tradition.tala_list

    def get_tala(self, tala_id):
        """Get specific information about a tala.

        :param tala_id: A tala id or uuid.
        :returns: uuid, name, artists, works, composers.
            ``artists`` includes artists with recording- and release-
            level relationships to a recording with this raga.
        """
        return self.tradition.get_taala(tala_id)

    def list_instruments(self):
        """List the instruments in the database. This function will automatically page through API results.

        :returns: A list of dictionaries containing instrument information:
            ``{"id": instrument id, "name": Name of the instrument}``
            For additional information about each instrument use :func:`get_instrument`.
        """
        return self.tradition.get_instruments()

    def get_instrument(self, instrument_id):
        """Get specific information about an instrument.

        :param instrument_id: An instrument id
        :returns: id, name, artists.
            ``artists`` includes artists with recording- and release-
            level performance relationships of this instrument.
        """
        return self.tradition.get_instrument(instrument_id)

    @staticmethod
    def list_available_types(recording_id):
        """Get the available source filetypes for a Musicbrainz recording.

        :param recording_id: Musicbrainz recording ID.
        :returns: a list of filetypes in the database for this recording.
        """
        document = dunya.conn._dunya_query_json("document/by-id/%s" % recording_id)
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
        return dunya.file_for_document(
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
        data = dunya.file_for_document(
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
        """Download the mp3 of a document and save it to the specified directory.

        :param recording_id: The MBID of the recording.
        :param output_dir: Where to save the mp3 to.
        :returns: name of the saved file.
        """
        return self.tradition.download_mp3(recording_id, output_dir)

    def download_concert(self, concert_id, output_dir):
        """Download the mp3s of all recordings in a concert and save them to the specified directory.

        :param concert_id: The MBID of the concert.
        :param location: Where to save the mp3s to.
        """
        return self.tradition.download_concert(concert_id, output_dir)
