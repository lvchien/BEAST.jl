include(joinpath(dirname(pathof(BEAST)),"../lib/gmsh.jl"))

using .gmsh
using CompScienceMeshes

"""
    meshsphere(radius, h)

Create a mesh of a sphere of radius `radius`

The target edge size is `h`.
"""
function meshsphere(radius, h)
    fno = tempname() * ".msh"

    gmsh.initialize()
    gmsh.model.add("sphere")

    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h, 1)
    gmsh.model.geo.addPoint(radius, 0.0, 0.0, h, 2)
    gmsh.model.geo.addPoint(0.0, radius, 0.0, h, 3)
    gmsh.model.geo.addCircleArc(2, 1, 3, 1)
    gmsh.model.geo.addPoint(-radius, 0.0, 0.0, h, 4)
    gmsh.model.geo.addPoint(0.0, -radius, 0.0, h, 5)
    gmsh.model.geo.addCircleArc(3, 1, 4, 2)
    gmsh.model.geo.addCircleArc(4, 1, 5, 3)
    gmsh.model.geo.addCircleArc(5, 1, 2, 4)
    gmsh.model.geo.addPoint(0.0, 0.0, -radius, h, 6)
    gmsh.model.geo.addPoint(0.0, 0.0, radius, h, 7)
    gmsh.model.geo.addCircleArc(3, 1, 6, 5)
    gmsh.model.geo.addCircleArc(6, 1, 5, 6)
    gmsh.model.geo.addCircleArc(5, 1, 7, 7)
    gmsh.model.geo.addCircleArc(7, 1, 3, 8)
    gmsh.model.geo.addCircleArc(2, 1, 7, 9)
    gmsh.model.geo.addCircleArc(7, 1, 4, 10)
    gmsh.model.geo.addCircleArc(4, 1, 6, 11)
    gmsh.model.geo.addCircleArc(6, 1, 2, 12)
    gmsh.model.geo.addCurveLoop([2, 8, -10], 13)
    gmsh.model.geo.addSurfaceFilling([13], 14)
    gmsh.model.geo.addCurveLoop([10, 3, 7], 15)
    gmsh.model.geo.addSurfaceFilling([15], 16)
    gmsh.model.geo.addCurveLoop([-8, -9, 1], 17)
    gmsh.model.geo.addSurfaceFilling([17], 18)
    gmsh.model.geo.addCurveLoop([-11, -2, 5], 19)
    gmsh.model.geo.addSurfaceFilling([19], 20)
    gmsh.model.geo.addCurveLoop([-5, -12, -1], 21)
    gmsh.model.geo.addSurfaceFilling([21], 22)
    gmsh.model.geo.addCurveLoop([-3, 11, 6], 23)
    gmsh.model.geo.addSurfaceFilling([23], 24)
    gmsh.model.geo.addCurveLoop([-7, 4, 9], 25)
    gmsh.model.geo.addSurfaceFilling([25], 26)
    gmsh.model.geo.addCurveLoop([-4, 12, -6], 27)
    gmsh.model.geo.addSurfaceFilling([27], 28)
    gmsh.model.geo.addSurfaceLoop([28, 26, 16, 14, 20, 24, 22, 18], 29)
    gmsh.model.geo.addVolume([29], 30)
    gmsh.model.geo.addPhysicalGroup(2, [28, 26, 16, 14, 20, 24, 22, 18], 1)
    gmsh.model.geo.addPhysicalGroup(3, [30], 2)

    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.MshFileVersion",2)
    gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()
    gmsh.write(fno)
    gmsh.finalize()

    m = CompScienceMeshes.read_gmsh_mesh(fno)
    rm(fno)
    return m

end


"""
    meshtorus(majorradius, minorradius, h)

Create a mesh of a torus of 2 radii `majorradius` and `minorradius`

The target edge size is `h`.
"""
function meshtorus(majorradius, minorradius, h)
    @assert minorradius < majorradius
    
    fno = tempname() * ".msh"

    gmsh.initialize()
    gmsh.model.add("torus")
    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h, 1)
    gmsh.model.geo.addPoint(0.0, 0.0, minorradius, h, 2)
    gmsh.model.geo.addPoint(0.0, 0.0, -minorradius, h, 3)

    gmsh.model.geo.addPoint(majorradius, 0.0, 0.0, h, 4)
    gmsh.model.geo.addPoint(majorradius + minorradius, 0.0, 0.0, h, 5)
    gmsh.model.geo.addPoint(majorradius, 0.0, minorradius, h, 6)
    gmsh.model.geo.addPoint(majorradius - minorradius, 0.0, 0.0, h, 7)
    gmsh.model.geo.addPoint(majorradius, 0.0, -minorradius, h, 8)
    gmsh.model.geo.addCircleArc(5, 4, 6, 1)
    gmsh.model.geo.addCircleArc(6, 4, 7, 2)
    gmsh.model.geo.addCircleArc(7, 4, 8, 3)
    gmsh.model.geo.addCircleArc(8, 4, 5, 4)

    gmsh.model.geo.addPoint(0.0, majorradius, 0.0, h, 9)
    gmsh.model.geo.addPoint(0.0, majorradius + minorradius, 0.0, h, 10)
    gmsh.model.geo.addPoint(0.0, majorradius, minorradius, h, 11)
    gmsh.model.geo.addPoint(0.0, majorradius - minorradius, 0.0, h, 12)
    gmsh.model.geo.addPoint(0.0, majorradius, -minorradius, h, 13)
    gmsh.model.geo.addCircleArc(10, 9, 11, 5)
    gmsh.model.geo.addCircleArc(11, 9, 12, 6)
    gmsh.model.geo.addCircleArc(12, 9, 13, 7)
    gmsh.model.geo.addCircleArc(13, 9, 10, 8)

    gmsh.model.geo.addPoint(-majorradius, 0.0, 0.0, h, 14)
    gmsh.model.geo.addPoint(-majorradius - minorradius, 0.0, 0.0, h, 15)
    gmsh.model.geo.addPoint(-majorradius, 0.0, minorradius, h, 16)
    gmsh.model.geo.addPoint(-majorradius + minorradius, 0.0, 0.0, h, 17)
    gmsh.model.geo.addPoint(-majorradius, 0.0, -minorradius, h, 18)
    gmsh.model.geo.addCircleArc(15, 14, 16, 9)
    gmsh.model.geo.addCircleArc(16, 14, 17, 10)
    gmsh.model.geo.addCircleArc(17, 14, 18, 11)
    gmsh.model.geo.addCircleArc(18, 14, 15, 12)

    gmsh.model.geo.addPoint(0.0, -majorradius, 0.0, h, 19)
    gmsh.model.geo.addPoint(0.0, -majorradius - minorradius, 0.0, h, 20)
    gmsh.model.geo.addPoint(0.0, -majorradius, minorradius, h, 21)
    gmsh.model.geo.addPoint(0.0, -majorradius + minorradius, 0.0, h, 22)
    gmsh.model.geo.addPoint(0.0, -majorradius, -minorradius, h, 23)
    gmsh.model.geo.addCircleArc(20, 19, 21, 13)
    gmsh.model.geo.addCircleArc(21, 19, 22, 14)
    gmsh.model.geo.addCircleArc(22, 19, 23, 15)
    gmsh.model.geo.addCircleArc(23, 19, 20, 16)
    
    gmsh.model.geo.addCircleArc(5, 1, 10, 17)
    gmsh.model.geo.addCircleArc(6, 2, 11, 18)
    gmsh.model.geo.addCircleArc(7, 1, 12, 19)
    gmsh.model.geo.addCircleArc(8, 3, 13, 20)

    gmsh.model.geo.addCircleArc(10, 1, 15, 21)
    gmsh.model.geo.addCircleArc(11, 2, 16, 22)
    gmsh.model.geo.addCircleArc(12, 1, 17, 23)
    gmsh.model.geo.addCircleArc(13, 3, 18, 24)

    gmsh.model.geo.addCircleArc(15, 1, 20, 25)
    gmsh.model.geo.addCircleArc(16, 2, 21, 26)
    gmsh.model.geo.addCircleArc(17, 1, 22, 27)
    gmsh.model.geo.addCircleArc(18, 3, 23, 28)

    gmsh.model.geo.addCircleArc(20, 1, 5, 29)
    gmsh.model.geo.addCircleArc(21, 2, 6, 30)
    gmsh.model.geo.addCircleArc(22, 1, 7, 31)
    gmsh.model.geo.addCircleArc(23, 3, 8, 32)

    gmsh.model.geo.addCurveLoop([-1, -18, 5, 17], 33)
    gmsh.model.geo.addSurfaceFilling([33], 1)
    gmsh.model.geo.addCurveLoop([-2, -19, 6, 18], 34)
    gmsh.model.geo.addSurfaceFilling([34], 2)
    gmsh.model.geo.addCurveLoop([-3, -20, 7, 19], 35)
    gmsh.model.geo.addSurfaceFilling([35], 3)
    gmsh.model.geo.addCurveLoop([-4, -17, 8, 20], 36)
    gmsh.model.geo.addSurfaceFilling([36], 4)

    gmsh.model.geo.addCurveLoop([-5, -22, 9, 21], 37)
    gmsh.model.geo.addSurfaceFilling([37], 5)
    gmsh.model.geo.addCurveLoop([-6, -23, 10, 22], 38)
    gmsh.model.geo.addSurfaceFilling([38], 6)
    gmsh.model.geo.addCurveLoop([-7, -24, 11, 23], 39)
    gmsh.model.geo.addSurfaceFilling([39], 7)
    gmsh.model.geo.addCurveLoop([-8, -21, 12, 24], 40)
    gmsh.model.geo.addSurfaceFilling([40], 8)
    
    gmsh.model.geo.addCurveLoop([-9, -26, 13, 25], 41)
    gmsh.model.geo.addSurfaceFilling([41], 9)
    gmsh.model.geo.addCurveLoop([-10, -27, 14, 26], 42)
    gmsh.model.geo.addSurfaceFilling([42], 10)
    gmsh.model.geo.addCurveLoop([-11, -28, 15, 27], 43)
    gmsh.model.geo.addSurfaceFilling([43], 11)
    gmsh.model.geo.addCurveLoop([-12, -25, 16, 28], 44)
    gmsh.model.geo.addSurfaceFilling([44], 12)

    gmsh.model.geo.addCurveLoop([-13, -30, 1, 29], 45)
    gmsh.model.geo.addSurfaceFilling([45], 13)
    gmsh.model.geo.addCurveLoop([-14, -31, 2, 30], 46)
    gmsh.model.geo.addSurfaceFilling([46], 14)
    gmsh.model.geo.addCurveLoop([-15, -32, 3, 31], 47)
    gmsh.model.geo.addSurfaceFilling([47], 15)
    gmsh.model.geo.addCurveLoop([-16, -29, 4, 32], 48)
    gmsh.model.geo.addSurfaceFilling([48], 16)

    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.MshFileVersion",2)
    gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()
    gmsh.write(fno)
    gmsh.finalize()

    m = CompScienceMeshes.read_gmsh_mesh(fno)
    rm(fno)
    return m
end



"""
    meshcuboid(width1, width2, height, h)

Create a mesh of a cuboid of size `width1 x width2 x height`.

The target edge size is `h`.
"""
function meshcuboid(width1, width2, height, h)    
    fno = tempname() * ".msh"
    gmsh.initialize()
    gmsh.model.add("cuboid")

    # bottom plate
    gmsh.model.geo.addPoint(width1/2, -width2/2, 0, h, 1)
    gmsh.model.geo.addPoint(width1/2, width2/2, 0, h, 2)
    gmsh.model.geo.addPoint(-width1/2, width2/2, 0, h, 3)
    gmsh.model.geo.addPoint(-width1/2, -width2/2, 0, h, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCurveLoop([-1, -2, -3, -4], 101)
    gmsh.model.geo.addPlaneSurface([101], 1)

    # top plate
    gmsh.model.geo.addPoint(width1/2, -width2/2, height, h, 5)
    gmsh.model.geo.addPoint(width1/2, width2/2, height, h, 6)
    gmsh.model.geo.addPoint(-width1/2, width2/2, height, h, 7)
    gmsh.model.geo.addPoint(-width1/2, -width2/2, height, h, 8)

    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 5, 8)
    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 102)    
    gmsh.model.geo.addPlaneSurface([102], 2)

    # sides
    gmsh.model.geo.addLine(1, 5, 9)
    gmsh.model.geo.addLine(2, 6, 10)
    gmsh.model.geo.addLine(3, 7, 11)
    gmsh.model.geo.addLine(4, 8, 12)

    gmsh.model.geo.addCurveLoop([1, -9,  -5, 10], 103)
    gmsh.model.geo.addPlaneSurface([103], 3)

    gmsh.model.geo.addCurveLoop([2, -10, -6, 11], 104)
    gmsh.model.geo.addPlaneSurface([104], 4)

    gmsh.model.geo.addCurveLoop([3, -11, -7, 12], 105)
    gmsh.model.geo.addPlaneSurface([105], 5)

    gmsh.model.geo.addCurveLoop([4, -12, -8, 9], 106)
    gmsh.model.geo.addPlaneSurface([106], 6)

    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.MshFileVersion",2)
    gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()
    gmsh.write(fno)
    gmsh.finalize()

    m = CompScienceMeshes.read_gmsh_mesh(fno)
    rm(fno)
    return m
end




"""
    meshsquaretorus(width, height, holewidth, h)

Create a mesh of a square torus of size `width` and `height` with a hole of size `holewidth` and `height`

The target edge size is `h`.
"""
function meshsquaretorus(width, height, holewidth, h)
    @assert holewidth < width
    
    fno = tempname() * ".msh"
    gmsh.initialize()
    gmsh.model.add("squaretorus")

    # bottom plate
    gmsh.model.geo.addPoint(holewidth/2, -holewidth/2, -height/2, h, 1)
    gmsh.model.geo.addPoint(holewidth/2, holewidth/2, -height/2, h, 2)
    gmsh.model.geo.addPoint(-holewidth/2, holewidth/2, -height/2, h, 3)
    gmsh.model.geo.addPoint(-holewidth/2, -holewidth/2, -height/2, h, 4)
    gmsh.model.geo.addPoint(width/2, -width/2, -height/2, h, 5)
    gmsh.model.geo.addPoint(width/2, width/2, -height/2, h, 6)
    gmsh.model.geo.addPoint(-width/2, width/2, -height/2, h, 7)
    gmsh.model.geo.addPoint(-width/2, -width/2, -height/2, h, 8)

    gmsh.model.geo.addLine(2, 3, 1)
    gmsh.model.geo.addLine(3, 4, 2)
    gmsh.model.geo.addLine(4, 1, 3)
    gmsh.model.geo.addLine(1, 2, 4)
    gmsh.model.geo.addCurveLoop([-1, -2, -3, -4], 101)

    gmsh.model.geo.addLine(6, 7, 5)
    gmsh.model.geo.addLine(7, 8, 6)
    gmsh.model.geo.addLine(8, 5, 7)
    gmsh.model.geo.addLine(5, 6, 8)
    gmsh.model.geo.addCurveLoop([-5, -6, -7, -8], 102)
    gmsh.model.geo.addPlaneSurface([-101, -102], 1)

    # top plate
    gmsh.model.geo.addPoint(holewidth/2, -holewidth/2, height/2, h, 9)
    gmsh.model.geo.addPoint(holewidth/2, holewidth/2, height/2, h, 10)
    gmsh.model.geo.addPoint(-holewidth/2, holewidth/2, height/2, h, 11)
    gmsh.model.geo.addPoint(-holewidth/2, -holewidth/2, height/2, h, 12)
    gmsh.model.geo.addPoint(width/2, -width/2, height/2, h, 13)
    gmsh.model.geo.addPoint(width/2, width/2, height/2, h, 14)
    gmsh.model.geo.addPoint(-width/2, width/2, height/2, h, 15)
    gmsh.model.geo.addPoint(-width/2, -width/2, height/2, h, 16)

    gmsh.model.geo.addLine(10, 11, 9)
    gmsh.model.geo.addLine(11, 12, 10)
    gmsh.model.geo.addLine(12, 9, 11)
    gmsh.model.geo.addLine(9, 10, 12)
    gmsh.model.geo.addCurveLoop([9, 10, 11, 12], 103)
    
    gmsh.model.geo.addLine(14, 15, 13)
    gmsh.model.geo.addLine(15, 16, 14)
    gmsh.model.geo.addLine(16, 13, 15)
    gmsh.model.geo.addLine(13, 14, 16)
    gmsh.model.geo.addCurveLoop([13, 14, 15, 16], 104)
    gmsh.model.geo.addPlaneSurface([-103, -104], 2)

    # sides
    gmsh.model.geo.addLine(2, 10, 17)
    gmsh.model.geo.addLine(3, 11, 18)
    gmsh.model.geo.addLine(4, 12, 19)
    gmsh.model.geo.addLine(1, 9, 20)
    gmsh.model.geo.addLine(6, 14, 21)
    gmsh.model.geo.addLine(7, 15, 22)
    gmsh.model.geo.addLine(8, 16, 23)
    gmsh.model.geo.addLine(5, 13, 24)

    gmsh.model.geo.addCurveLoop([-12, 17, 4, -20], 105)
    gmsh.model.geo.addPlaneSurface([-105], 3)

    gmsh.model.geo.addCurveLoop([1, 18, -9, -17], 106)
    gmsh.model.geo.addPlaneSurface([-106], 4)

    gmsh.model.geo.addCurveLoop([-18, -10, 19, 2], 107)
    gmsh.model.geo.addPlaneSurface([-107], 5)

    gmsh.model.geo.addCurveLoop([-19, -11, 20, 3], 108)
    gmsh.model.geo.addPlaneSurface([-108], 6)

    gmsh.model.geo.addCurveLoop([24, 16, -21, -8], 109)
    gmsh.model.geo.addPlaneSurface([-109], 7)

    gmsh.model.geo.addCurveLoop([-5, -22, 13, 21], 110)
    gmsh.model.geo.addPlaneSurface([-110], 8)

    gmsh.model.geo.addCurveLoop([-6, -23, 14, 22], 111)
    gmsh.model.geo.addPlaneSurface([-111], 9)

    gmsh.model.geo.addCurveLoop([23, 15, -24, -7], 112)
    gmsh.model.geo.addPlaneSurface([-112], 10)

    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.MshFileVersion",2)
    gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()
    gmsh.write(fno)
    gmsh.finalize()

    m = CompScienceMeshes.read_gmsh_mesh(fno)
    rm(fno)
    return m
end


"""
    meshsquaretorus4holes(width, height, holewidth, h)

Create a mesh of a square torus of size `width` and `height` with 4 holes of size `holewidth` and `height`

The target edge size is `h`.
"""

function meshsquaretorus4holes(width, height, holewidth, h)
    @assert 2*holewidth < width
    
    fno = tempname() * ".msh"
    gmsh.initialize()
    gmsh.model.add("squaretorus4holes")

    # bottom plate
    gmsh.model.geo.addPoint(width/4 + holewidth/2, -width/4 - holewidth/2, -height/2, h, 1)
    gmsh.model.geo.addPoint(width/4 + holewidth/2, -width/4 + holewidth/2, -height/2, h, 2)
    gmsh.model.geo.addPoint(width/4 - holewidth/2, -width/4 + holewidth/2, -height/2, h, 3)
    gmsh.model.geo.addPoint(width/4 - holewidth/2, -width/4 - holewidth/2, -height/2, h, 4)

    gmsh.model.geo.addPoint(width/4 + holewidth/2, width/4 - holewidth/2, -height/2, h, 5)
    gmsh.model.geo.addPoint(width/4 + holewidth/2, width/4 + holewidth/2, -height/2, h, 6)
    gmsh.model.geo.addPoint(width/4 - holewidth/2, width/4 + holewidth/2, -height/2, h, 7)
    gmsh.model.geo.addPoint(width/4 - holewidth/2, width/4 - holewidth/2, -height/2, h, 8)

    gmsh.model.geo.addPoint(-width/4 + holewidth/2, width/4 - holewidth/2, -height/2, h, 9)
    gmsh.model.geo.addPoint(-width/4 + holewidth/2, width/4 + holewidth/2, -height/2, h, 10)
    gmsh.model.geo.addPoint(-width/4 - holewidth/2, width/4 + holewidth/2, -height/2, h, 11)
    gmsh.model.geo.addPoint(-width/4 - holewidth/2, width/4 - holewidth/2, -height/2, h, 12)

    gmsh.model.geo.addPoint(-width/4 + holewidth/2, -width/4 - holewidth/2, -height/2, h, 13)
    gmsh.model.geo.addPoint(-width/4 + holewidth/2, -width/4 + holewidth/2, -height/2, h, 14)
    gmsh.model.geo.addPoint(-width/4 - holewidth/2, -width/4 + holewidth/2, -height/2, h, 15)
    gmsh.model.geo.addPoint(-width/4 - holewidth/2, -width/4 - holewidth/2, -height/2, h, 16)

    gmsh.model.geo.addPoint(width/2, -width/2, -height/2, h, 17)
    gmsh.model.geo.addPoint(width/2, width/2, -height/2, h, 18)
    gmsh.model.geo.addPoint(-width/2, width/2, -height/2, h, 19)
    gmsh.model.geo.addPoint(-width/2, -width/2, -height/2, h, 20)

    gmsh.model.geo.addLine(2, 3, 1)
    gmsh.model.geo.addLine(3, 4, 2)
    gmsh.model.geo.addLine(4, 1, 3)
    gmsh.model.geo.addLine(1, 2, 4)
    gmsh.model.geo.addCurveLoop([-1, -2, -3, -4], 101)

    gmsh.model.geo.addLine(6, 7, 5)
    gmsh.model.geo.addLine(7, 8, 6)
    gmsh.model.geo.addLine(8, 5, 7)
    gmsh.model.geo.addLine(5, 6, 8)
    gmsh.model.geo.addCurveLoop([-5, -6, -7, -8], 102)

    gmsh.model.geo.addLine(10, 11, 9)
    gmsh.model.geo.addLine(11, 12, 10)
    gmsh.model.geo.addLine(12, 9, 11)
    gmsh.model.geo.addLine(9, 10, 12)
    gmsh.model.geo.addCurveLoop([-9, -10, -11, -12], 103)

    gmsh.model.geo.addLine(14, 15, 13)
    gmsh.model.geo.addLine(15, 16, 14)
    gmsh.model.geo.addLine(16, 13, 15)
    gmsh.model.geo.addLine(13, 14, 16)
    gmsh.model.geo.addCurveLoop([-13, -14, -15, -16], 104)

    gmsh.model.geo.addLine(18, 19, 17)
    gmsh.model.geo.addLine(19, 20, 18)
    gmsh.model.geo.addLine(20, 17, 19)
    gmsh.model.geo.addLine(17, 18, 20)
    gmsh.model.geo.addCurveLoop([-17, -18, -19, -20], 105)

    gmsh.model.geo.addPlaneSurface([-101, -102, -103, -104, -105], 1)

    # top plate
    gmsh.model.geo.addPoint(width/4 + holewidth/2, -width/4 - holewidth/2, height/2, h, 21)
    gmsh.model.geo.addPoint(width/4 + holewidth/2, -width/4 + holewidth/2, height/2, h, 22)
    gmsh.model.geo.addPoint(width/4 - holewidth/2, -width/4 + holewidth/2, height/2, h, 23)
    gmsh.model.geo.addPoint(width/4 - holewidth/2, -width/4 - holewidth/2, height/2, h, 24)

    gmsh.model.geo.addPoint(width/4 + holewidth/2, width/4 - holewidth/2, height/2, h, 25)
    gmsh.model.geo.addPoint(width/4 + holewidth/2, width/4 + holewidth/2, height/2, h, 26)
    gmsh.model.geo.addPoint(width/4 - holewidth/2, width/4 + holewidth/2, height/2, h, 27)
    gmsh.model.geo.addPoint(width/4 - holewidth/2, width/4 - holewidth/2, height/2, h, 28)

    gmsh.model.geo.addPoint(-width/4 + holewidth/2, width/4 - holewidth/2, height/2, h, 29)
    gmsh.model.geo.addPoint(-width/4 + holewidth/2, width/4 + holewidth/2, height/2, h, 30)
    gmsh.model.geo.addPoint(-width/4 - holewidth/2, width/4 + holewidth/2, height/2, h, 31)
    gmsh.model.geo.addPoint(-width/4 - holewidth/2, width/4 - holewidth/2, height/2, h, 32)

    gmsh.model.geo.addPoint(-width/4 + holewidth/2, -width/4 - holewidth/2, height/2, h, 33)
    gmsh.model.geo.addPoint(-width/4 + holewidth/2, -width/4 + holewidth/2, height/2, h, 34)
    gmsh.model.geo.addPoint(-width/4 - holewidth/2, -width/4 + holewidth/2, height/2, h, 35)
    gmsh.model.geo.addPoint(-width/4 - holewidth/2, -width/4 - holewidth/2, height/2, h, 36)

    gmsh.model.geo.addPoint(width/2, -width/2, height/2, h, 37)
    gmsh.model.geo.addPoint(width/2, width/2, height/2, h, 38)
    gmsh.model.geo.addPoint(-width/2, width/2, height/2, h, 39)
    gmsh.model.geo.addPoint(-width/2, -width/2, height/2, h, 40)

    gmsh.model.geo.addLine(22, 23, 21)
    gmsh.model.geo.addLine(23, 24, 22)
    gmsh.model.geo.addLine(24, 21, 23)
    gmsh.model.geo.addLine(21, 22, 24)
    gmsh.model.geo.addCurveLoop([-21, -22, -23, -24], 106)

    gmsh.model.geo.addLine(26, 27, 25)
    gmsh.model.geo.addLine(27, 28, 26)
    gmsh.model.geo.addLine(28, 25, 27)
    gmsh.model.geo.addLine(25, 26, 28)
    gmsh.model.geo.addCurveLoop([-25, -26, -27, -28], 107)

    gmsh.model.geo.addLine(30, 31, 29)
    gmsh.model.geo.addLine(31, 32, 30)
    gmsh.model.geo.addLine(32, 29, 31)
    gmsh.model.geo.addLine(29, 30, 32)
    gmsh.model.geo.addCurveLoop([-29, -30, -31, -32], 108)

    gmsh.model.geo.addLine(34, 35, 33)
    gmsh.model.geo.addLine(35, 36, 34)
    gmsh.model.geo.addLine(36, 33, 35)
    gmsh.model.geo.addLine(33, 34, 36)
    gmsh.model.geo.addCurveLoop([-33, -34, -35, -36], 109)

    gmsh.model.geo.addLine(38, 39, 37)
    gmsh.model.geo.addLine(39, 40, 38)
    gmsh.model.geo.addLine(40, 37, 39)
    gmsh.model.geo.addLine(37, 38, 40)
    gmsh.model.geo.addCurveLoop([-37, -38, -39, -40], 110)

    gmsh.model.geo.addPlaneSurface([106, 107, 108, 109, 110], 2)

    # sides
    gmsh.model.geo.addLine(2, 22, 41)
    gmsh.model.geo.addLine(3, 23, 42)
    gmsh.model.geo.addLine(4, 24, 43)
    gmsh.model.geo.addLine(1, 21, 44)
    gmsh.model.geo.addLine(6, 26, 45)
    gmsh.model.geo.addLine(7, 27, 46)
    gmsh.model.geo.addLine(8, 28, 47)
    gmsh.model.geo.addLine(5, 25, 48)
    gmsh.model.geo.addLine(10, 30, 49)
    gmsh.model.geo.addLine(11, 31, 50)
    gmsh.model.geo.addLine(12, 32, 51)
    gmsh.model.geo.addLine(9, 29, 52)
    gmsh.model.geo.addLine(14, 34, 53)
    gmsh.model.geo.addLine(15, 35, 54)
    gmsh.model.geo.addLine(16, 36, 55)
    gmsh.model.geo.addLine(13, 33, 56)
    gmsh.model.geo.addLine(18, 38, 57)
    gmsh.model.geo.addLine(19, 39, 58)
    gmsh.model.geo.addLine(20, 40, 59)
    gmsh.model.geo.addLine(17, 37, 60)

    gmsh.model.geo.addCurveLoop([1, 42, -21, -41], 111)
    gmsh.model.geo.addPlaneSurface([-111], 3)

    gmsh.model.geo.addCurveLoop([2, 43, -22, -42], 112)
    gmsh.model.geo.addPlaneSurface([-112], 4)

    gmsh.model.geo.addCurveLoop([3, 44, -23, -43], 113)
    gmsh.model.geo.addPlaneSurface([-113], 5)

    gmsh.model.geo.addCurveLoop([4, 41, -24, -44], 114)
    gmsh.model.geo.addPlaneSurface([-114], 6)


    gmsh.model.geo.addCurveLoop([5, 46, -25, -45], 115)
    gmsh.model.geo.addPlaneSurface([-115], 7)

    gmsh.model.geo.addCurveLoop([6, 47, -26, -46], 116)
    gmsh.model.geo.addPlaneSurface([-116], 8)

    gmsh.model.geo.addCurveLoop([7, 48, -27, -47], 117)
    gmsh.model.geo.addPlaneSurface([-117], 9)

    gmsh.model.geo.addCurveLoop([8, 45, -28, -48], 118)
    gmsh.model.geo.addPlaneSurface([-118], 10)
    

    gmsh.model.geo.addCurveLoop([9, 50, -29, -49], 119)
    gmsh.model.geo.addPlaneSurface([-119], 11)

    gmsh.model.geo.addCurveLoop([10, 51, -30, -50], 120)
    gmsh.model.geo.addPlaneSurface([-120], 12)

    gmsh.model.geo.addCurveLoop([11, 52, -31, -51], 121)
    gmsh.model.geo.addPlaneSurface([-121], 13)

    gmsh.model.geo.addCurveLoop([12, 49, -32, -52], 122)
    gmsh.model.geo.addPlaneSurface([-122], 14)

    
    gmsh.model.geo.addCurveLoop([13, 54, -33, -53], 123)
    gmsh.model.geo.addPlaneSurface([-123], 15)

    gmsh.model.geo.addCurveLoop([14, 55, -34, -54], 124)
    gmsh.model.geo.addPlaneSurface([-124], 16)

    gmsh.model.geo.addCurveLoop([15, 56, -35, -55], 125)
    gmsh.model.geo.addPlaneSurface([-125], 17)

    gmsh.model.geo.addCurveLoop([16, 53, -36, -56], 126)
    gmsh.model.geo.addPlaneSurface([-126], 18)

    
    gmsh.model.geo.addCurveLoop([17, 58, -37, -57], 127)
    gmsh.model.geo.addPlaneSurface([127], 19)

    gmsh.model.geo.addCurveLoop([18, 59, -38, -58], 128)
    gmsh.model.geo.addPlaneSurface([128], 20)

    gmsh.model.geo.addCurveLoop([19, 60, -39, -59], 129)
    gmsh.model.geo.addPlaneSurface([129], 21)

    gmsh.model.geo.addCurveLoop([20, 57, -40, -60], 130)
    gmsh.model.geo.addPlaneSurface([130], 22)

    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.MshFileVersion",2)
    gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()
    gmsh.write(fno)
    gmsh.finalize()

    m = CompScienceMeshes.read_gmsh_mesh(fno)
    rm(fno)
    return m
end