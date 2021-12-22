class TrueModel:
    def out(self, z):
        return -3.5 * z**2 + 3.6 * z - 0.1
    def err(self, z, z_err):
        return 0
