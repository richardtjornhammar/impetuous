"""
Copyright 2026 RICHARD TJÖRNHAMMAR

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

class SpiralingEquations:
    """
    Curved LaTeX-style equations on an expanding Archimedean spiral.

    Install:
        python -m pip install matplotlib

    Creates vector PDF/SVG and a high-resolution PNG.

    Each equation is converted to vector outlines, then its glyphs are
    warped along the local curvature of the spiral. No external LaTeX
    installation is required (Matplotlib mathtext is used).
    """
    def __init__(
        self,
        equations           = [(r"$x^2+y^2=1$", 14), (r"$x^2+y^2=1$", 14)],
        basename            = "equation_spiral",
        figure_size_inches  = 14,
        dpi                 = 500,
        centre_radius       = 2.25,
        start_radius_offset = 0.22,
        pitch_per_turn      = 0.90,
        equation_gap_inches = 0.12,
        font_size           = 11,
        curvature_strength  = 1.0,
        flip_curvature      = False,
        ink                 = "#172b55",
        paper               = "white",
        start_angle_degrees = 90,
        draw_centre_circle  = False,
        show_start_label    = False,
        start_label         = "Start",
        font                = None,
        output_dir          = None,
    ):
        self.EQUATIONS           = equations
        self.OUTPUT_DIR          = self.set_param( Path("."), output_dir )
        self.BASENAME            = self.set_param( "equation_spiral", basename )
        self.FIGURE_SIZE_INCHES  = self.set_param( 14 , figure_size_inches )
        self.DPI                 = self.set_param( 500, dpi )
        self.CENTRE_RADIUS       = self.set_param( 2.25, centre_radius )
        self.START_RADIUS        = self.CENTRE_RADIUS + self.set_param( 0.22, start_radius_offset )
        self.PITCH_PER_TURN      = self.set_param( 0.90 , pitch_per_turn )
        self.EQUATION_GAP_INCHES = self.set_param( 0.12 , equation_gap_inches )
        self.FONT_SIZE           = self.set_param( 11 , font_size )
        # Curvature multiplier:
        #   1.0 = follows the local curvature
        #   0.0 = straight equations (only rotated to follow the tangent)
        #   1.5 = more pronounced bending
        self.CURVATURE_STRENGTH  = self.set_param( 1.0 , curvature_strength )
        # Default curvature side for all equations.
        # False = bend toward the empty centre of the spiral.
        # True  = bend away from the empty centre.
        # Individual equations can override this with a third tuple value.
        self.FLIP_CURVATURE      = self.set_param( False, flip_curvature )
        self.INK                 = self.set_param( "#172b55", ink )
        self.PAPER               = self.set_param( "white" , paper )
        self.START_ANGLE_DEGREES = self.set_param( 90 , start_angle_degrees )
        # Set to False to leave the centre completely blank (no circle outline).
        self.DRAW_CENTRE_CIRCLE  = self.set_param( False, draw_centre_circle )
        self.SHOW_START_LABEL    = self.set_param( True, show_start_label )
        self.FONT                = self.set_param( FontProperties(family="STIXGeneral") , font )
        self.START_LABEL         = start_label

    def set_param(self, default, value ):
        result = default if value is None else value
        return result


    def unpack_equation(self,item):
        """Return (mathtext, font_size, flip_curvature) for an equation entry."""
        if isinstance(item, tuple):
            if len(item) == 2:
                expression, size = item
                flip = self.FLIP_CURVATURE
            elif len(item) == 3:
                expression, size, flip = item
            else:
                raise ValueError(
                    "Equation tuples must be (expression, size) or "
                    "(expression, size, flip_curvature)."
                )
            return expression, float(size), bool(flip)
        return item, float(self.FONT_SIZE), bool(self.FLIP_CURVATURE)


    def curved_text_path(self, expression, fontsize, x0, y0, tangent_angle,
                        curvature, data_units_per_point, flip_curvature=False):
        """
        Convert one mathtext expression to vector outlines and bend its
        baseline along a circular approximation to the spiral at (x0, y0).

        The approximation uses the spiral's local curvature. Each point in
        the glyph outlines is mapped from local tangent/normal coordinates
        onto the curved baseline, so individual characters bend smoothly.
        """
        text_path = TextPath(
            (0, 0), expression, size=fontsize, prop=self.FONT,
            usetex=False
        )

        vertices = text_path.vertices.copy()
        codes = text_path.codes

        if len(vertices) == 0:
            return None

        bounds = text_path.get_extents()
        # Centre the complete equation around its anchor point.
        vertices[:, 0] -= 0.5 * (bounds.x0 + bounds.x1)
        vertices[:, 1] -= 0.5 * (bounds.y0 + bounds.y1)

        # TextPath coordinates are points. Convert them to spiral data units.
        local_x = vertices[:, 0] * data_units_per_point
        local_y = vertices[:, 1] * data_units_per_point

        # The signed curvature of the outward clockwise spiral bends the
        # baseline away from the central opening. Reverse its sign by default
        # so equations bow toward the opening; flip_curvature restores the
        # opposite direction for an individual equation.
        bend_sign = 1.0 if flip_curvature else -1.0
        k = bend_sign * curvature * self.CURVATURE_STRENGTH
        if abs(k) < 1e-10:
            along = local_x
            normal = local_y
        else:
            angle = k * local_x
            sin_a = [math.sin(v) for v in angle]
            cos_a = [math.cos(v) for v in angle]

            # Circular-arc centreline plus the glyph's normal offset.
            along = [math.sin(k * x) / k - y * math.sin(k * x)
                    for x, y in zip(local_x, local_y)]
            normal = [(1.0 - math.cos(k * x)) / k + y * math.cos(k * x)
                    for x, y in zip(local_x, local_y)]

        phi = math.radians(tangent_angle)
        tx, ty = math.cos(phi), math.sin(phi)
        nx, ny = -ty, tx

        mapped = [
            (x0 + a * tx + n * nx, y0 + a * ty + n * ny)
            for a, n in zip(along, normal)
        ]

        return MplPath(mapped, codes)


    def advance_u_by_arc_length(self, u, distance, b):
        """Advance along the Archimedean spiral by a requested arc distance."""
        if distance <= 0:
            return u

        # ds/du = sqrt(r(u)^2 + b^2), with r(u) = START_RADIUS + b*u.
        # Midpoint iteration accounts for the slowly increasing spiral radius.
        du = distance / math.sqrt((self.START_RADIUS + b * u) ** 2 + b ** 2)
        for _ in range(6):
            r_mid = self.START_RADIUS + b * (u + 0.5 * du)
            du = distance / math.sqrt(r_mid * r_mid + b * b)
        return u + du

    def make_spiral(self):
        if not self.EQUATIONS:
            raise ValueError("Add at least one equation to EQUATIONS.")

        b = self.PITCH_PER_TURN / (2.0 * math.pi)
        theta0 = math.radians(self.START_ANGLE_DEGREES)

        # Estimate a generous canvas radius. The actual spiral end is
        # determined dynamically by equation lengths below.
        limit = self.START_RADIUS + max(3.0, len(self.EQUATIONS) / 5) * self.PITCH_PER_TURN + 2.0

        fig, ax = plt.subplots(figsize=(self.FIGURE_SIZE_INCHES,
                                        self.FIGURE_SIZE_INCHES))
        fig.patch.set_facecolor(self.PAPER)
        ax.set_facecolor(self.PAPER)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

        if self.DRAW_CENTRE_CIRCLE:
            ax.add_patch(Circle(
                (0, 0), self.CENTRE_RADIUS, fill=False,
                linewidth=0.8, edgecolor=self.INK, zorder=1
            ))

        if self.SHOW_START_LABEL:
            ax.text(
                0, limit * 0.91, self.START_LABEL,
                ha="center", va="center", fontsize=self.FONT_SIZE,
                color=self.INK, fontfamily="STIXGeneral"
            )

        fig.subplots_adjust(left=0.015, right=0.985,
                            bottom=0.015, top=0.94)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_bbox = ax.get_window_extent(renderer=renderer)
        axes_width_inches = axes_bbox.width / fig.dpi
        data_units_per_inch = (2.0 * limit) / axes_width_inches
        data_units_per_point = data_units_per_inch / 72.0
        gap_units = self.EQUATION_GAP_INCHES * data_units_per_inch

        # Measure every equation before positioning any of them. Since each
        # equation is centred on its anchor, the distance between two anchors
        # must use HALF of each neighbouring equation's width.
        measured = []
        for item in self.EQUATIONS:
            expression, fontsize, flip_curvature = self.unpack_equation(item)
            text_path = TextPath(
                (0, 0), expression, size=fontsize, prop=self.FONT, usetex=False
            )
            bounds = text_path.get_extents()
            measured.append((expression, fontsize, flip_curvature,
                            text_path, bounds, bounds.width))

        u = 0.0
        for index, (expression, fontsize, flip_curvature,
                    text_path, bounds, text_width_points) in enumerate(measured):

            radius = self.START_RADIUS + b * u
            theta = theta0 - u

            x0 = radius * math.cos(theta)
            y0 = radius * math.sin(theta)

            # First and second derivatives of the clockwise spiral.
            dx = b * math.cos(theta) + radius * math.sin(theta)
            dy = b * math.sin(theta) - radius * math.cos(theta)
            ddx = radius * math.cos(theta)
            ddy = radius * math.sin(theta)

            tangent_angle = math.degrees(math.atan2(dy, dx))
            speed_sq = dx * dx + dy * dy
            # Signed curvature: its sign carries the actual bending direction.
            # The helper applies the inward/outward choice via flip_curvature.
            curvature = (dx * ddy - dy * ddx) / (speed_sq ** 1.5)

            text_width_units = text_width_points * data_units_per_point

            curved_path = self.curved_text_path(
                expression, fontsize, x0, y0,
                tangent_angle, curvature, data_units_per_point,
                flip_curvature=flip_curvature
            )
            if curved_path is not None:
                ax.add_patch(PathPatch(
                    curved_path, facecolor=self.INK, edgecolor="none", zorder=3
                ))

            # Centre-to-centre distance = half this equation + half the next
            # equation + the requested clear gap. This avoids overlap when a
            # short equation is followed by a much longer one, and avoids the
            # excessive spacing caused by advancing a full previous width.
            if index + 1 < len(measured):
                next_width_units = measured[index + 1][5] * data_units_per_point
                centre_spacing = (
                    0.5 * text_width_units
                    + 0.5 * next_width_units
                    + gap_units
                )
                u = self.advance_u_by_arc_length(u, centre_spacing, b)

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = self.OUTPUT_DIR / f"{self.BASENAME}.pdf"
        svg_path = self.OUTPUT_DIR / f"{self.BASENAME}.svg"
        png_path = self.OUTPUT_DIR / f"{self.BASENAME}.png"

        fig.savefig(pdf_path, facecolor=self.PAPER, bbox_inches="tight")
        fig.savefig(svg_path, facecolor=self.PAPER, bbox_inches="tight")
        fig.savefig(png_path, dpi=self.DPI, facecolor=self.PAPER, bbox_inches="tight")
        plt.close(fig)

        print(f"Created:\n  {pdf_path}\n  {svg_path}\n  {png_path}")


if __name__ == "__main__":

    """
    OUTPUT_DIR = Path(".")
    BASENAME = "equation_spiral"
    FIGURE_SIZE_INCHES = 14
    DPI = 500
    CENTRE_RADIUS = 2.25
    START_RADIUS = CENTRE_RADIUS + 0.22
    PITCH_PER_TURN = 0.90
    # Gap between consecutive equations, measured along the spiral.
    EQUATION_GAP_INCHES = 0.12
    # Used when an equation entry is a plain string.
    FONT_SIZE = 11
    # Curvature multiplier:
    #   1.0 = follows the local curvature
    #   0.0 = straight equations (only rotated to follow the tangent)
    #   1.5 = more pronounced bending
    CURVATURE_STRENGTH = 1.0
    # Default curvature side for all equations.
    # False = bend toward the empty centre of the spiral.
    # True  = bend away from the empty centre.
    # Individual equations can override this with a third tuple value.
    FLIP_CURVATURE = False
    INK = "#172b55"
    PAPER = "white"
    START_ANGLE_DEGREES = 90
    # Set to False to leave the centre completely blank (no circle outline).
    DRAW_CENTRE_CIRCLE = False
    SHOW_START_LABEL = True
    FONT = FontProperties(family="STIXGeneral")
    """

    esize = 30
    equations = [
        ( (
        r"$F(f(\rho)) = 4\pi F_0 \int_{R/\xi}^{\infty}"
        r"\left[f^2 + \gamma_0\left(\frac{df}{d\rho}\right)^2"
        r"+ \left(\frac{1}{\rho}\frac{d}{d\rho}"
        r"\rho\frac{d}{d\rho} f \right)^2\right]\rho d\rho \quad , $"
        ) , esize),
        ( (r"$\rho^4\frac{d^4 f}{d\rho^4}"
        r"+ 2 \rho^3\frac{d^3 f}{d\rho^3}"
        r" - (1-\gamma_0\rho^2)"
        r"\left(\rho^2\frac{d^2 f}{d\rho^2} "
                r"- \rho\frac{df}{d\rho}\right)"
        r" + \rho^4 f = 0 \quad , $"), esize),
        (r"$\lim_{\gamma_0\to 0 }  \quad , $", esize),
        ((r"$ \hspace{2} \left( \frac{d^2}{d\rho^2} + \frac{1}{\rho}\frac{d}{d\rho} - i \right)"
        r"\left( \frac{d^2}{d\rho^2} + \frac{1}{\rho}\frac{d}{d\rho} + i \right)f = 0 \quad , $"), esize),
        ((r"$f(\rho) = a_0 \Re[J_0(\rho e^{\pi i / 4})]"
        r"+ a_1 \Im[J_0(\rho e^{\pi i / 4})]"
        r"+ b_0 \Re[K_0(\rho e^{\pi i / 4})]"
        r"+ b_1 \Im[K_0(\rho e^{\pi i / 4})] = "
        r"a_0 g_b + a_1 \hat{g}_b + b_0 g_k + b_1 \hat{g}_k \quad , $"), esize),
        (r"$n(\rho) = \hat{g}' g - \hat{g} g' \quad , $", esize),
        (r"$\frac{1}{\rho}\frac{d}{d\rho}\rho\frac{d}{d\rho}\hat{g} = g \quad , $", esize),
        (r"$\frac{1}{\rho}\frac{d}{d\rho}\rho\frac{d}{d\rho}g = - \hat{g} \quad , $", esize),
        (r"$F[f] = 4 \pi F_0 \left(b_0^2+b_1^2\right)\int_{\Omega} [\hat{g}_k^2+g_k^2]\rho d\rho \quad , $", esize),
        (r"$F[f] = 4 \pi F_0 \frac{R}{\xi} \left[ S ( b_0^2 + b_1^2 ) N\right] \quad , $", esize),
        (r"$N = n(\frac{R}{\xi})  \quad , $", esize),
        (r"$S ( b_0^2 + b_1^2 ) N\rightarrow 1/\sqrt{2} \quad \text{so} \quad , $", esize),
        (r"$\gamma_L = \sqrt{2} F_0/\xi $", esize)
    ]
    """
• family="STIXNonUnicode"
• family="DejaVu Serif"
• family="Liberation Serif"
• family="cmr10" (Computer Modern Roman—the classic LaTeX font)
    """
    se = SpiralingEquations( equations = equations,
                             font = FontProperties(family="cmr10"),
                             font_size = esize)
    se.make_spiral()
