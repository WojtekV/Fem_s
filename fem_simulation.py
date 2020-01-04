import numpy as np
from grid import Grid
from universal_element import UniversalElement
import shape_f


class FEM:
    def __init__(self, height, width, nodes_height, nodes_width):
        self.g = Grid(height, width, nodes_height, nodes_width)
        self.u_e = UniversalElement()

        self.weights_hbc = [1, 1]
        self.weights = [1, 1, 1, 1]
        self.size_of_jakob = 2
        self.vector_columns = 1
        self.number_of_sides = 4
        self.number_of_pc_for_bc = 2
        self.dn_dx_dn_dy_mat_rows = 2

        self.k = 25
        self.dT = 50
        self.cw = 700
        self.ro = 7800
        self.alfa = 300
        self.t_ot = 1200
        self.simulation_time = 500
        self.simulation_time_step = 50
        self.number_of_steps = int(self.simulation_time / self.simulation_time_step)

        self.jakobians = np.zeros((self.g.n_e, self.u_e.num_of_pc, self.size_of_jakob, self.size_of_jakob))
        self.inverse_jakobians = np.zeros((self.g.n_e, self.u_e.num_of_pc, self.size_of_jakob, self.size_of_jakob))
        self.dn_dx_dn_dy = np.zeros((self.g.n_e, self.u_e.num_of_pc, self.u_e.num_of_pc, self.dn_dx_dn_dy_mat_rows,
                                     self.vector_columns))
        self.calc_jakobians()
        self.calculate_inverse_jakobians()
        self.calculate_dx_dx_dn_dy_2x1()

        self.h_4x1 = np.zeros((self.g.n_e, self.u_e.num_of_pc, self.dn_dx_dn_dy_mat_rows,self.u_e.num_of_shape_f,
                               self.vector_columns))
        self.h_4x4_for_pc = np.zeros((self.g.n_e, self.u_e.num_of_pc, self.u_e.num_of_shape_f, self.u_e.num_of_shape_f))
        self.h_for_el = np.zeros((self.g.n_e, self.u_e.num_of_shape_f, self.u_e.num_of_shape_f))
        self.calculate_h_4x1()
        self.calculate_h_4x4_for_pc()
        self.calculate_h_for_elements()

        self.c_4x1 = np.zeros((self.u_e.num_of_pc, self.u_e.num_of_shape_f, self.vector_columns))
        self.c_4x4_for_pc = np.zeros((self.g.n_e, self.u_e.num_of_pc, self.u_e.num_of_shape_f, self.u_e.num_of_shape_f))
        self.c_for_el = np.zeros((self.g.n_e, self.u_e.num_of_shape_f, self.u_e.num_of_shape_f))
        self.calculate_c_4x1()
        self.calculate_c_4x4_for_pc()
        self.calculate_c_for_elements()

        self.hbc_4x1 = np.zeros((self.number_of_sides, self.number_of_pc_for_bc, self.u_e.num_of_shape_f,
                                 self.vector_columns))
        self.hbc_4x4 = np.zeros((self.number_of_sides, self.u_e.num_of_shape_f, self.u_e.num_of_shape_f))
        self.calculate_hbc_4x1()
        self.calculate_hbc_4x4()

        self.h_plus_hbc = np.zeros((self.g.n_e, self.u_e.num_of_shape_f, self.u_e.num_of_shape_f))
        self.calculate_h_plus_hbc()

        self.H_g = np.zeros((self.g.n_n, self.g.n_n))
        self.C_g = np.zeros((self.g.n_n, self.g.n_n))
        self.H_Cdt = np.zeros((self.g.n_n, self.g.n_n))
        self.calculate_h_g_c_g()
        self.calculate_h_cdt_g()

        self.p_for_side_pc = np.zeros((self.number_of_sides, self.number_of_pc_for_bc, self.u_e.num_of_shape_f,
                                       self.vector_columns))
        self.p_for_side = np.zeros((self.number_of_sides, self.u_e.num_of_shape_f, self.vector_columns))
        self.p_for_elements = np.zeros((self.g.n_e, self.u_e.num_of_shape_f, self.vector_columns))
        self.calculate_p_for_side_pc()
        self.calculate_p_for_sides()
        self.calculate_p_for_elements()

        self.P_g = np.zeros((self.g.n_n, 1))
        self.calculate_p_g()

        self.T_1 = np.zeros((self.g.n_n, 1))
        self.T_1.fill(100)
        self.P_Cdt_T0 = np.zeros((self.g.n_n, 1))
        self.calculate_p_cdt_t0()

        self.min_max_temperatures = np.zeros((self.number_of_steps, 1, 2))
        self.simulate()

    # TODO: shorten the code using nested loops
    def calc_jakobians(self):
        for i in range(self.g.n_e):
            for j in range(self.u_e.num_of_pc):
                jak = np.zeros((self.size_of_jakob, self.size_of_jakob))
                for k in range(self.u_e.num_of_shape_f):
                    jak[0][0] += self.u_e.dn_dksi[j][k] * self.g.nodes[self.g.elements[i].nodes_id[k]].x
                    jak[0][1] += self.u_e.dn_dksi[j][k] * self.g.nodes[self.g.elements[i].nodes_id[k]].y
                    jak[1][0] += self.u_e.dn_deta[j][k] * self.g.nodes[self.g.elements[i].nodes_id[k]].x
                    jak[1][1] += self.u_e.dn_deta[j][k] * self.g.nodes[self.g.elements[i].nodes_id[k]].y
                self.jakobians[i][j] = jak

    def calculate_inverse_jakobians(self):
        for i in range(self.g.n_e):
            for j in range(self.u_e.num_of_pc):
                self.inverse_jakobians[i][j] = np.linalg.inv(self.jakobians[i][j])

    def calculate_dx_dx_dn_dy_2x1(self):
        for i in range(self.g.n_e):
            for j in range(self.u_e.num_of_pc):
                for k in range(self.u_e.num_of_shape_f):
                    self.dn_dx_dn_dy[i][j][k] = self.inverse_jakobians[i][k].dot(self.u_e.dn_dksi_dn_deta[j][k])

    def calculate_h_4x1(self):
        for i in range(self.g.n_e):
            for j in range(self.u_e.num_of_pc):
                # 0 is dx, 1 is dy
                for k in range(self.dn_dx_dn_dy_mat_rows):
                    for l in range(self.u_e.num_of_shape_f):
                        self.h_4x1[i][j][k][l] = self.dn_dx_dn_dy[i][j][l][k]

    def calculate_h_4x4_for_pc(self):
        for i in range(self.g.n_e):
            for j in range(self.u_e.num_of_pc):
                self.h_4x4_for_pc[i][j] = self.h_4x1[i][j][0].dot(self.h_4x1[i][j][0].transpose()) + \
                                          self.h_4x1[i][j][1].dot(self.h_4x1[i][j][1].transpose())

    def calculate_h_for_elements(self):
        for i in range(self.g.n_e):
            h = np.zeros((self.u_e.num_of_shape_f, self.u_e.num_of_shape_f))
            for j in range(self.u_e.num_of_shape_f):
                h += self.h_4x4_for_pc[i][j] * self.weights[j] * np.linalg.det(self.jakobians[i][j])
            self.h_for_el[i] = h * self.k

    def calculate_c_4x1(self):
        for i in range(self.u_e.num_of_pc):
            for j in range(self.u_e.num_of_shape_f):
                self.c_4x1[i][j] = self.u_e.n[i][j]

    def calculate_c_4x4_for_pc(self):
        for i in range(self.g.n_e):
            for j in range(self.u_e.num_of_pc):
                self.c_4x4_for_pc[i][j] = self.c_4x1[j].dot(self.c_4x1[j].transpose()) * \
                                          np.linalg.det(self.jakobians[i][j]) * self.cw * self.ro

    def calculate_c_for_elements(self):
        for i in range(self.g.n_e):
            for j in range(self.u_e.num_of_pc):
                self.c_for_el[i] += self.c_4x4_for_pc[i][j]

    # TODO: shorten the code using nested loops
    def calculate_hbc_4x1(self):
        # MATRIX side / point / shape_f
        # 1:
        self.hbc_4x1[0][0][0] = shape_f.n1(self.u_e.int_points_for_Hbc[0][0][0], self.u_e.int_points_for_Hbc[0][0][1])
        self.hbc_4x1[0][0][1] = shape_f.n2(self.u_e.int_points_for_Hbc[0][0][0], self.u_e.int_points_for_Hbc[0][0][1])

        self.hbc_4x1[0][1][0] = shape_f.n1(self.u_e.int_points_for_Hbc[0][1][0], self.u_e.int_points_for_Hbc[0][1][1])
        self.hbc_4x1[0][1][1] = shape_f.n2(self.u_e.int_points_for_Hbc[0][1][0], self.u_e.int_points_for_Hbc[0][1][1])
        # 2:
        self.hbc_4x1[1][0][1] = shape_f.n2(self.u_e.int_points_for_Hbc[1][0][0], self.u_e.int_points_for_Hbc[1][0][1])
        self.hbc_4x1[1][0][2] = shape_f.n3(self.u_e.int_points_for_Hbc[1][0][0], self.u_e.int_points_for_Hbc[1][0][1])

        self.hbc_4x1[1][1][1] = shape_f.n2(self.u_e.int_points_for_Hbc[1][1][0], self.u_e.int_points_for_Hbc[1][1][1])
        self.hbc_4x1[1][1][2] = shape_f.n3(self.u_e.int_points_for_Hbc[1][1][0], self.u_e.int_points_for_Hbc[1][1][1])
        # 3:
        self.hbc_4x1[2][0][2] = shape_f.n3(self.u_e.int_points_for_Hbc[2][0][0], self.u_e.int_points_for_Hbc[2][0][1])
        self.hbc_4x1[2][0][3] = shape_f.n4(self.u_e.int_points_for_Hbc[2][0][0], self.u_e.int_points_for_Hbc[2][0][1])

        self.hbc_4x1[2][1][2] = shape_f.n3(self.u_e.int_points_for_Hbc[2][1][0], self.u_e.int_points_for_Hbc[2][1][1])
        self.hbc_4x1[2][1][3] = shape_f.n4(self.u_e.int_points_for_Hbc[2][1][0], self.u_e.int_points_for_Hbc[2][1][1])
        # 4:
        self.hbc_4x1[3][0][0] = shape_f.n1(self.u_e.int_points_for_Hbc[3][0][0], self.u_e.int_points_for_Hbc[3][0][1])
        self.hbc_4x1[3][0][3] = shape_f.n4(self.u_e.int_points_for_Hbc[3][0][0], self.u_e.int_points_for_Hbc[3][0][1])

        self.hbc_4x1[3][1][0] = shape_f.n1(self.u_e.int_points_for_Hbc[3][1][0], self.u_e.int_points_for_Hbc[3][1][1])
        self.hbc_4x1[3][1][3] = shape_f.n4(self.u_e.int_points_for_Hbc[3][1][0], self.u_e.int_points_for_Hbc[3][1][1])

    def calculate_hbc_4x4(self):
        for i in range(0, 4, 2):
            self.hbc_4x4[i] = (self.hbc_4x1[i][0].dot(self.hbc_4x1[i][0].transpose()) * self.weights_hbc[0] +
                               self.hbc_4x1[i][1].dot(self.hbc_4x1[i][1].transpose()) * self.weights_hbc[1]) * \
                               self.alfa * self.g.dx / 2
            self.hbc_4x4[i + 1] = (self.hbc_4x1[i + 1][0].dot(self.hbc_4x1[i + 1][0].transpose()) *
                                   self.weights_hbc[0] +
                                   self.hbc_4x1[i + 1][1].dot(self.hbc_4x1[i + 1][1].transpose()) *
                                   self.weights_hbc[1]) * self.alfa * self.g.dy / 2

    def calculate_h_plus_hbc(self):
        for i in range(self.g.n_e):
            self.h_plus_hbc[i] = np.copy(self.h_for_el[i])
        j = self.g.n_e - 1
        for i in range(self.g.n_h - 1):
            self.h_plus_hbc[i] += self.hbc_4x4[3]
            self.h_plus_hbc[j] += self.hbc_4x4[1]
            j -= 1
        j = self.g.n_h - 2
        i = 0
        while i < self.g.n_e - 1:
            self.h_plus_hbc[i] += self.hbc_4x4[0]
            self.h_plus_hbc[j] += self.hbc_4x4[2]
            i += self.g.n_h - 1
            j += self.g.n_h - 1

    def calculate_h_g_c_g(self):
        for i in range(self.g.n_e):
            i1 = 0
            for j in self.g.elements[i].nodes_id:
                i2 = 0
                for k in self.g.elements[i].nodes_id:
                    self.H_g[j][k] += self.h_plus_hbc[i][i1][i2]
                    self.C_g[j][k] += self.c_for_el[i][i1][i2]
                    i2 += 1
                i1 += 1

    def calculate_h_cdt_g(self):
        self.H_Cdt = self.H_g + self.C_g / self.dT

    # TODO: shorten the code using nested loops
    def calculate_p_for_side_pc(self):
        # POINT:   side / point / coordinate
        # MATRIX: side / point / shape_f
        # 1:
        self.p_for_side_pc[0][0][0] = -shape_f.n1(self.u_e.int_points_for_Hbc[0][0][0],
                                                  self.u_e.int_points_for_Hbc[0][0][1])
        self.p_for_side_pc[0][0][1] = -shape_f.n2(self.u_e.int_points_for_Hbc[0][0][0],
                                                  self.u_e.int_points_for_Hbc[0][0][1])

        self.p_for_side_pc[0][1][0] = -shape_f.n1(self.u_e.int_points_for_Hbc[0][1][0],
                                                  self.u_e.int_points_for_Hbc[0][1][1])
        self.p_for_side_pc[0][1][1] = -shape_f.n2(self.u_e.int_points_for_Hbc[0][1][0],
                                                  self.u_e.int_points_for_Hbc[0][1][1])
        # 2:
        self.p_for_side_pc[1][0][1] = -shape_f.n2(self.u_e.int_points_for_Hbc[1][0][0],
                                                  self.u_e.int_points_for_Hbc[1][0][1])
        self.p_for_side_pc[1][0][2] = -shape_f.n3(self.u_e.int_points_for_Hbc[1][0][0],
                                                  self.u_e.int_points_for_Hbc[1][0][1])

        self.p_for_side_pc[1][1][1] = -shape_f.n2(self.u_e.int_points_for_Hbc[1][1][0],
                                                  self.u_e.int_points_for_Hbc[1][1][1])
        self.p_for_side_pc[1][1][2] = -shape_f.n3(self.u_e.int_points_for_Hbc[1][1][0],
                                                  self.u_e.int_points_for_Hbc[1][1][1])
        # 3:
        self.p_for_side_pc[2][0][2] = -shape_f.n3(self.u_e.int_points_for_Hbc[2][0][0],
                                                  self.u_e.int_points_for_Hbc[2][0][1])
        self.p_for_side_pc[2][0][3] = -shape_f.n4(self.u_e.int_points_for_Hbc[2][0][0],
                                                  self.u_e.int_points_for_Hbc[2][0][1])

        self.p_for_side_pc[2][1][2] = -shape_f.n3(self.u_e.int_points_for_Hbc[2][1][0],
                                                  self.u_e.int_points_for_Hbc[2][1][1])
        self.p_for_side_pc[2][1][3] = -shape_f.n4(self.u_e.int_points_for_Hbc[2][1][0],
                                                  self.u_e.int_points_for_Hbc[2][1][1])
        # 4:
        self.p_for_side_pc[3][0][0] = -shape_f.n1(self.u_e.int_points_for_Hbc[3][0][0],
                                                  self.u_e.int_points_for_Hbc[3][0][1])
        self.p_for_side_pc[3][0][3] = -shape_f.n4(self.u_e.int_points_for_Hbc[3][0][0],
                                                  self.u_e.int_points_for_Hbc[3][0][1])

        self.p_for_side_pc[3][1][0] = -shape_f.n1(self.u_e.int_points_for_Hbc[3][1][0],
                                                  self.u_e.int_points_for_Hbc[3][1][1])
        self.p_for_side_pc[3][1][3] = -shape_f.n4(self.u_e.int_points_for_Hbc[3][1][0],
                                                  self.u_e.int_points_for_Hbc[3][1][1])

    def calculate_p_for_sides(self):
        for i in range(0, 4, 2):
            self.p_for_side[i] = (self.hbc_4x1[i][0] * self.weights_hbc[0] +
                                  self.hbc_4x1[i][1] * self.weights_hbc[1]) * \
                                  self.alfa * self.t_ot * self.g.dx / 2
            self.p_for_side[i + 1] = (self.hbc_4x1[i + 1][0] * self.weights_hbc[0] +
                                      self.hbc_4x1[i + 1][1] * self.weights_hbc[1]) * \
                                      self.alfa * self.t_ot * self.g.dy / 2

    def calculate_p_for_elements(self):
        j = self.g.n_e - 1
        for i in range(self.g.n_h - 1):
            self.p_for_elements[i] += self.p_for_side[3]
            self.p_for_elements[j] += self.p_for_side[1]
            j -= 1
        j = self.g.n_h - 2
        i = 0
        while i < self.g.n_e - 1:
            self.p_for_elements[i] += self.p_for_side[0]
            self.p_for_elements[j] += self.p_for_side[2]
            i += self.g.n_h - 1
            j += self.g.n_h - 1

    def calculate_p_g(self):
        for i in range(self.g.n_e):
            i1 = 0
            for j in self.g.elements[i].nodes_id:
                self.P_g[j] += self.p_for_elements[i][i1]
                i1 += 1

    def calculate_p_cdt_t0(self):
        self.P_Cdt_T0 = self.P_g + (self.C_g/self.dT).dot(self.T_1)

    def simulate(self):
        for i in range(10):
            self.T_1 = np.linalg.inv(self.H_Cdt).dot((self.C_g/self.dT).dot(self.T_1) + self.P_g)
            self.min_max_temperatures[i] = [self.T_1.min(), self.T_1.max()]


f = FEM(0.1, 0.1, 4, 4)
print(f.min_max_temperatures)




