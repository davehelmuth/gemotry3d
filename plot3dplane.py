"""
This script defines and demonstrates the use of classes for representing and 
manipulating a plane in 3D space, utilizing both general and computational 
forms of the plane's equation.
"""

import math
import numpy as np
import numpy.typing as npt
from typing import TypeAlias
import matplotlib.pyplot as plt

from dataclasses import dataclass, field


@dataclass
class Point:
    """A point in 3D space
    
    Args:
        x: The x coordinate of the point. Defaults to 0.0.
        y: The y coordinate of the point. Defaults to 0.0.
        z: The z coordinate of the point. Defaults to 0.0.
    """

    x: float = field(default=0.0)
    """The x coordinate of the point"""

    y: float = field(default=0.0)
    """The y coordinate of the point"""

    z: float = field(default=0.0)
    """The z coordinate of the point"""



@dataclass
class PlaneGeneralForm:
    """Represents the equation of plane in General Form.
    
    Args:
        a: The constant a. Default 0.0
        b: The constant b. Default 0.0
        c: The constant c. Must be more than zero  Default 0.01
        known_point: A known point on the plane. Default Point(x=0.0, y=0.0, z=0.0)

    Note: 
        The General Form Equation of a plane is:
        
            a(x - x1) + b(y - y1) + c(z - z1) = 0

        - where:
            - a, b, c are constants of which c is never equal to zero
            - x1, y1, z1 represent a known point on the plane
            - x, y, z represents any unknown point on the plane
    
    Raises:
        ValueError: Constant c can not be within 1e-9 of zero
    """

    a: float = field(default=0.0)
    """The constant a"""

    b: float = field(default=0.0)
    """The constant b"""

    c: float = field(default=0.01)
    """The constant c"""

    known_point: Point = field(default_factory=lambda: Point(x=0.0, y=0.0, z=0.0))
    """A known point on the plane."""

    def __post_init__(self):
        if math.isclose(self.c, 0.0, abs_tol=1e-9):
            raise ValueError("Coefficient 'c' cannot be close to zero.")

    @property
    def x1(self) -> float:
        """The x-coordinate from the known point on the plane."""
        return self.known_point.x
    
    @property
    def y1(self) -> float:
        """The y-coordinate from the known point on the plane."""
        return self.known_point.y

    @property
    def z1(self) -> float:
        """The z-coordinate from the known point on the plane."""

        return self.known_point.z


XYPairs: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
"""
A tuple of two 2D NumPy arrays representing the X and Y coordinate pairs 
that make up a mesh grid.

This type alias is used to represent a pair of 2D NumPy arrays where:
- The first element in the Tuple is the X coordinates array.
- The second element in the Tuple is the Y coordinates array.

The arrays are typically generated using `numpy.meshgrid` from two 1D arrays 
created by `numpy.linspace`. These pairs of X and Y coordinates can be used 
for further calculations, such as evaluating a function Z = f(X, Y) over a 
grid in a 3D space.

Examples:
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y)
    xy_pair: XYPairs = (X, Y)
"""


class PlaneComputationalForm:
    """Represents the equation of plane in Computational Form.
    
    To be utilized by software, the equation of a plane must be in this form. 

    Args:
        general_form: The equation of a plane in general form

    Note:
        The equation in computational form is:

            Z = aX + bY + c

        - where:
            - a, b, c are constants. 
                - These are not the same values from the general equation.
            - X, Y each pair calculates a unique value of Z 

    """

    def __init__(self, a: float, b: float, c: float) -> None:
        self.a = a
        self.b = b
        self.c = c

    @classmethod
    def init_from_general_form(cls, gen_equation: PlaneGeneralForm) -> 'PlaneComputationalForm':
        """Initialize an instance of PlaneComputationalForm by transforming an 
        instance PlaneGeneralForm.

        Args:
            gen_equation: The general form of the plane equation.

        Note:
            - This is an alternative constructor to be used when
        """

        a = -1.0 * (gen_equation.a / gen_equation.c)
        b = -1.0 * (gen_equation.b / gen_equation.c)

        addend0 = (gen_equation.a * gen_equation.x1) / gen_equation.c
        addend1 = (gen_equation.b * gen_equation.y1) / gen_equation.c
        addend2 = gen_equation.z1
        c = addend0 + addend1 + addend2

        return cls(a, b, c)
    
    @staticmethod
    def generate_xy_pairs() -> XYPairs:
        """Generate 400 X, Y pairs that can be used to calculate z values
        
        Returns:
            A mesh like 2D grid of x,y pairs to be used to calculate the Z 
            values for a function (i.e. plane) in 3D space.
        """
        x = np.linspace(-10, 10, 20)
        y = np.linspace(-10, 10, 20)
        X, Y = np.meshgrid(x, y)

        return X, Y

    def calculate_z_values(self, mesh_grid: XYPairs) -> npt.NDArray[np.float64]:
        """Calculate Z Values from meshgrid.
        
        Args:
            mesh_grid: The x,y value pairs used to calculate z

        Returns:
            np.ndarray: Computed Z values.
        """

        X, Y = mesh_grid

        # Computationally friendly version of plane equation
        Z = self.a * X + self.b * Y + self.c

        return Z

    def __repr__(self) -> str:
        return f'PlaneComputationalForm(a={self.a}, b={self.b}, c={self.c})'


def plot_plane(plane: PlaneComputationalForm) -> None:
    """Plot the plane in 3D space
    
    Args:
        plane: The plane to be plotted in 3D space
    """

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mesh_grid = plane.generate_xy_pairs()
    Z = plane.calculate_z_values(mesh_grid)

    X, Y = mesh_grid
    ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, color='b')  # type: ignore

    # Labels and titles
    ax.set_xlabel('X coordinates')
    ax.set_ylabel('Y coordinates')
    ax.set_zlabel('Z coordinates')    # type: ignore
    ax.set_title('3D Plane from equation z = ax + by + c')

    # Annotating the four corners
    
    corners = [(-10, -10), (-10, 10), (10, -10), (10, 10)]
    for cx, cy in corners:
        cz = plane.a * cx + plane.b * cy + plane.c
        ax.text(cx, cy, cz, f"({cx}, {cy}, {cz:.1f})", color='red', fontsize=12)  # type: ignore

    # Show the plot
    plt.show()


def main():

    #  a(x - x1) + b(y - y1) + c(z - z1) = 0

    myplane = PlaneGeneralForm(a=1.0, b=0.0, c=0.001)

    myplane3d = PlaneComputationalForm.init_from_general_form(myplane)

    plot_plane(myplane3d)


if __name__ == '__main__': main()
