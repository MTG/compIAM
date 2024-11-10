from compiam.io import (
    write_csv,
    write_json,
    write_scalar_txt,
)
from compmusic import dunya  # Using compmusic API to access the Dunya database
from compmusic.dunya import carnatic, hindustani

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

        if tradition not in ["carnatic", "hindustani"]:
            raise ValueError(
                "Please choose a valid tradition: carnatic or hindustani"
            )
        self.tradition = carnatic if tradition == "carnatic" else hindustani

        # Functions from the compmusic API are added as a method in the Corpora class
        for name in dir(self.tradition):
            func = getattr(self.tradition, name)
            if callable(func):
                setattr(self, name, func)

        logger.warning("""
            Note that a part of the collection is under restricted access.
            To access the full collection please request permission at https://dunya.compmusic.upf.edu/user/profile/
        """)

    def get_collection(self, recording_detail=False):
        """Get the documents (recordings) in a collection.

        :param recording_detail: Include additional information about each recording.
        :returns: dictionary with the recordings in the collection.
        """
        if recording_detail is False:
            logger.info(
                "To parse the entire collection with all recording details, "
                + "please use the .get_collection(recording_detail=True) method. "
                + "Please note that it might take a few moments..."
            )
        return self.tradition.get_recordings(recording_detail)
    
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
