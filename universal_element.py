import numpy as np
import math
import shape_f


class UniversalElement:
    def __init__(self):
        self.ksi_axis = 0
        self.eta_axis = 1
        self.num_of_pc = 4
        self.num_of_shape_f = 4
        self.dn_dksi_dn_deta_col = 1
        self.dn_dksi_dn_deta_rows = 2

        self.integration_points_for_elem = []
        self.set_integration_points()

        self.int_points_for_Hbc = []
        self.set_integration_points_for_hbc()

        self.dn_dksi = np.zeros((self.num_of_pc, self.num_of_pc))
        self.set_dn_dksi_matrix()

        self.dn_deta = np.zeros((self.num_of_pc, self.num_of_pc))
        self.set_dn_deta_matrix()

        self.n = np.zeros((self.num_of_pc, self.num_of_pc))
        self.set_n_px_matrix()

        self.dn_dksi_dn_deta = np.zeros(
            (self.num_of_pc, self.num_of_shape_f, self.dn_dksi_dn_deta_rows, self.dn_dksi_dn_deta_col))
        self.calculate_dn_dksi_dn_deta_matrices()

    # integration points used in gauss method (values ​​from the tables)
    def set_integration_points(self):
        self.integration_points_for_elem = [
            [-1 / math.sqrt(3), -1 / math.sqrt(3)],
            [1 / math.sqrt(3), -1 / math.sqrt(3)],
            [1 / math.sqrt(3), 1 / math.sqrt(3)],
            [-1 / math.sqrt(3), 1 / math.sqrt(3)]
        ]

    # integration points used in gauss method for boundary condition derivatives (values ​​from the tables)
    # side / point / axis
    def set_integration_points_for_hbc(self):
        self.int_points_for_Hbc = [
            [[-1/math.sqrt(3), -1], [1/math.sqrt(3), -1]],
            [[1, -1/math.sqrt(3)], [1, 1/math.sqrt(3)]],
            [[1/math.sqrt(3), 1], [-1/math.sqrt(3), 1]],
            [[-1, 1/math.sqrt(3)], [-1, -1/math.sqrt(3)]]
        ]

    def set_dn_dksi_matrix(self):
        for i in range(self.num_of_pc):
            for j in range(self.num_of_shape_f):
                self.dn_dksi[i][j] = shape_f.dn_dksi[j](self.integration_points_for_elem[i][self.eta_axis])

    def set_dn_deta_matrix(self):
        for i in range(self.num_of_pc):
            for j in range(self.num_of_shape_f):
                self.dn_deta[i][j] = shape_f.dn_deta[j](self.integration_points_for_elem[i][self.ksi_axis])

    def set_n_px_matrix(self):
        for i in range(self.num_of_pc):
            for j in range(self.num_of_shape_f):
                self.n[i][j] = shape_f.n[j](
                    self.integration_points_for_elem[i][self.ksi_axis],
                    self.integration_points_for_elem[i][self.eta_axis])

    # 2x1 matrices for each integration point (2 for each point (dx, dy)) (facilitates subsequent calculations)
    def calculate_dn_dksi_dn_deta_matrices(self):
        for i in range(self.num_of_pc):
            for j in range(self.num_of_shape_f):
                # 0 below is first row, 1 is second row
                self.dn_dksi_dn_deta[i][j][0] = self.dn_dksi[i][j]
                self.dn_dksi_dn_deta[i][j][1] = self.dn_deta[i][j]
