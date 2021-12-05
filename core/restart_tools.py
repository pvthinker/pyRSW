import glob
import os
import pickle


class Restart():
    """
    Tools to write/read restart from within the model (not offline)
    """

    def __init__(self, param, grid):
        self.param = param
        self.grid = grid

        self.myrank = grid.myrank
        self.output_dir = os.path.expanduser(
            f"{param.datadir}/{param.expname}")
        self.current_index = self._get_current_index()

    def _get_current_index(self):
        restarts = glob.glob(f"{self.output_dir}/restart_*")
        return len(restarts)

    @property
    def previous_index(self):
        return self.current_index-1

    def directory(self, index):
        return f"{self.output_dir}/restart_{index:02}"

    def filename(self, index):
        fname = f"restart_{index:02}_{self.myrank:04}.pkl"
        return fname

    def read(self):
        assert self.previous_index >= 0
        fdir = self.directory(self.previous_index)
        fname = self.filename(self.previous_index)
        pklfile = f"{fdir}/{fname}"
        assert os.path.isfile(pklfile)
        with open(pklfile, "rb") as fid:
            infos = pickle.load(fid)
        if self.myrank == 0:
            print(f"  Read {fname}")
        return infos

    def write(self, infos):
        fdir = self.directory(self.current_index)
        os.makedirs(fdir)
        fname = self.filename(self.current_index)
        pklfile = f"{fdir}/{fname}"
        with open(pklfile, "wb") as fid:
            pickle.dump(infos, fid)
        #print(f"[INFO] write {fname}")
