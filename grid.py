from node import Node
from element import Element


class Grid:
    def __init__(self, height, width, nodes_height, nodes_width):
        self.h = height
        self.w = width
        self.n_h = nodes_height
        self.n_w = nodes_width
        self.n_n = self.n_h * self.n_w
        self.n_e = (self.n_h - 1) * (self.n_w - 1)
        self.dx = self.w / (self.n_w - 1)
        self.dy = self.h / (self.n_h - 1)

        self.nodes = []
        self.create_nodes()

        self.elements = []
        self.create_elements()

    def create_elements(self):
        var = 0
        i = 0
        for i in range(self.n_e):
            if i % (self.n_h - 1) == 0 and i != 0:
                var += 1
            self.elements.append(
                Element(i + var, i + self.n_h + var, i + self.n_h + 1 + var, i + 1 + var))

    def create_nodes(self):
        x_pos = 0
        y_pos = 0
        for i in range(self.n_n):
            # creating nodes and setting their positions
            self.nodes.append(Node(i, x_pos, y_pos))
            y_pos += self.dy
            if y_pos > self.h:
                y_pos = 0
                x_pos += self.dx

        # setting boundary conditions
        # vertically
        j = self.n_n - 1
        for i in range(self.n_h):
            self.nodes[i].bc = True
            self.nodes[j].bc = True
            j -= 1
        # horizontally
        j = self.n_h - 1
        i = 0
        while i < self.n_n:
            self.nodes[i].bc = True
            self.nodes[j].bc = True
            i += self.n_h
            j += self.n_h
