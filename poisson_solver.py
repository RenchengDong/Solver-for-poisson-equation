#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rencheng
A solver for the 2D Poisson Equation
Physical model: Darcy flow equation in a 2D porous medium
Solution is the 2D fluid pressure field
Numerical method: finite volume method
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm


class PoissonSolver(object):
    def __init__(self, LX, LY, NX, NY):
        # LX: domain length along X coordinate
        # LY: domain length along Y coordinate
        # NX: the number of grids along X coordinate
        # NY: the number of grids along Y coordinate
        self.LX = LX
        self.LY = LY
        self.NX = NX
        self.NY = NY
        # NGrid: the number of total elements
        self.NGrid = self.NX * self.NY
        # perm is the permeability field and can be set heterogeneous
        self.perm = 1.0 * np.ones((self.NGrid, 2))
        # Thickness: thickness of porous medium along Z coordinate
        self.Thickness = 1.0
        # mu: fluid viscosity
        self.mu = 1.0
        # dX: mesh size along X coordinate
        # dY: mesh size along Y coordinate
        self.dX = self.LX / self.NX
        self.dY = self.LY / self.NY
        self.x_array = np.linspace(self.dX / 2, self.LX - self.dX / 2, self.NX)
        self.y_array = np.linspace(self.dY / 2, self.LY - self.dY / 2, self.NY)
        self.X_plot, self.Y_plot = np.meshgrid(self.x_array, self.y_array)
        self.X_coor = np.reshape(self.X_plot, (self.NGrid, 1))
        self.Y_coor = np.reshape(self.Y_plot, (self.NGrid, 1))
        self.AX = self.dY * self.Thickness * np.ones(self.NGrid)
        self.AY = self.dX * self.Thickness * np.ones(self.NGrid)
        self.dV = self.dX * self.dY * self.Thickness * np.ones(self.NGrid)
        # BC_type: type of boundary conditions; zero means Dirichlet boundary
        self.BC_type = np.zeros(4)
        # BC_value: pressure value at Dirichlet boundary
        self.BC_value = np.zeros(4)
        self.LHS = np.zeros((self.NGrid, self.NGrid))
        self.RHS = np.zeros(self.NGrid)

    # Thalf: calculate transmissibility between elements
    def Thalf(self, i, j, boundary_marker):
        # permeability is harmonic average of two neighboring elements
        delta = abs(i - j)
        if delta == 1 or boundary_marker == 1:
            Xperm_area = 2.0 / (
                1.0 / (self.perm[i, 0] * self.AX[i])
                + 1.0 / (self.perm[j, 0] * self.AX[j])
            )
            T = (Xperm_area) / (self.mu * self.dX)
        elif delta == self.NX or boundary_marker == 2:
            Yperm_area = 2.0 / (
                1.0 / (self.perm[i, 1] * self.AY[i])
                + 1.0 / (self.perm[j, 1] * self.AY[j])
            )
            T = (Yperm_area) / (self.mu * self.dY)
        return T

    # Source: calculate source term
    def Source(self, i):
        x = self.X_coor[i]
        y = self.Y_coor[i]
        return 2 * x * (1 - x) + 2 * y * (1 - y)

    # Assemble: assemble the linear system
    def Assemble(self):
        for l in range(self.NGrid):
            if (l + 1) % self.NX != 1:
                boundary_marker = 0
                self.LHS[l, l - 1] = -self.Thalf(l, l - 1, boundary_marker)
                self.LHS[l, l] -= self.LHS[l, l - 1]
            elif self.BC_type[0] == 0:
                # handle the left boundary
                boundary_marker = 1
                T_boundary = 2 * self.Thalf(l, l, boundary_marker)
                self.LHS[l, l] += T_boundary
                self.RHS[l] += T_boundary * self.BC_value[0]

            if (l + 1) % self.NX != 0:
                boundary_marker = 0
                self.LHS[l, l + 1] = -self.Thalf(l, l + 1, boundary_marker)
                self.LHS[l, l] -= self.LHS[l, l + 1]
            elif self.BC_type[2] == 0:
                # handle the right boundary
                boundary_marker = 1
                T_boundary = 2 * self.Thalf(l, l, boundary_marker)
                self.LHS[l, l] += T_boundary
                self.RHS[l] += T_boundary * self.BC_value[2]

            if math.ceil((l + 1) / self.NX) > 1:
                boundary_marker = 0
                self.LHS[l, l - self.NX] = -self.Thalf(l, l - self.NX, boundary_marker)
                self.LHS[l, l] -= self.LHS[l, l - self.NX]
            elif self.BC_type[1] == 0:
                # handle the bottom boundary
                boundary_marker = 2
                T_boundary = 2 * self.Thalf(l, l, boundary_marker)
                self.LHS[l, l] += T_boundary
                self.RHS[l] += T_boundary * self.BC_value[1]

            if math.ceil((l + 1) / self.NX) < self.NY:
                boundary_marker = 0
                self.LHS[l, l + self.NX] = -self.Thalf(l, l + self.NX, boundary_marker)
                self.LHS[l, l] -= self.LHS[l, l + self.NX]
            elif self.BC_type[3] == 0:
                # handle the top boundary
                boundary_marker = 2
                T_boundary = 2 * self.Thalf(l, l, boundary_marker)
                self.LHS[l, l] += T_boundary
                self.RHS[l] += T_boundary * self.BC_value[3]

            self.RHS[l] += self.Source(l).item() * self.dV[l]

    # Solve: solve the linear system
    def Solve(self):
        self.sol = np.linalg.solve(self.LHS, self.RHS)

    # Run: driver code for this solver
    def Run(self):
        self.Assemble()
        self.Solve()

    # PlotSol: plot the pressure solution
    def PlotSol(self):
        Z = np.reshape(self.sol, (self.NX, self.NY))
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        fig_num = ax.plot_surface(
            self.X_plot,
            self.Y_plot,
            Z,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        fig.colorbar(fig_num, shrink=0.5, aspect=5)
        plt.show()


# example problem to solve
my_solver = PoissonSolver(1, 1, 10, 10)
my_solver.Run()
my_solver.PlotSol()
