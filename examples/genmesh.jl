include("../lib/gmsh.jl")

using .gmsh
using CompScienceMeshes

function meshsphere2(radius, h)
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

meshsphere2(;radius, h) = meshsphere2(radius, h)

"""
    meshtorus(innerradius, outerradius, h)
    meshtorus(;innerradius, outerradius, h)

Create a mesh of a torus of 2 radii `innerradius` and `outerradius`

The target edge size is `h`.
"""
function meshtorus(innerradius, outerradius, h)
    @assert innerradius < outerradius
    
    fno = tempname() * ".msh"
    center = (outerradius + innerradius)/2
    radius = (outerradius - innerradius)/2

    gmsh.initialize()
    gmsh.model.add("torus")
    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h, 1)
    gmsh.model.geo.addPoint(0.0, 0.0, radius, h, 2)
    gmsh.model.geo.addPoint(0.0, 0.0, -radius, h, 3)

    # 1st quarter
    gmsh.model.geo.addPoint(center, 0.0, 0.0, h, 4)
    gmsh.model.geo.addPoint(outerradius, 0.0, 0.0, h, 5)
    gmsh.model.geo.addPoint(center, 0.0, radius, h, 6)
    gmsh.model.geo.addPoint(innerradius, 0.0, 0.0, h, 7)
    gmsh.model.geo.addPoint(center, 0.0, -radius, h, 8)
    gmsh.model.geo.addCircleArc(5, 4, 6, 1)
    gmsh.model.geo.addCircleArc(6, 4, 7, 2)
    gmsh.model.geo.addCircleArc(7, 4, 8, 3)
    gmsh.model.geo.addCircleArc(8, 4, 5, 4)

    gmsh.model.geo.addPoint(0.0, center, 0.0, h, 9)
    gmsh.model.geo.addPoint(0.0, outerradius, 0.0, h, 10)
    gmsh.model.geo.addPoint(0.0, center, radius, h, 11)
    gmsh.model.geo.addPoint(0.0, innerradius, 0.0, h, 12)
    gmsh.model.geo.addPoint(0.0, center, -radius, h, 13)
    gmsh.model.geo.addCircleArc(10, 9, 11, 5)
    gmsh.model.geo.addCircleArc(11, 9, 12, 6)
    gmsh.model.geo.addCircleArc(12, 9, 13, 7)
    gmsh.model.geo.addCircleArc(13, 9, 10, 8)
    
    gmsh.model.geo.addCircleArc(5, 1, 10, 9)
    gmsh.model.geo.addCircleArc(6, 2, 11, 10)
    gmsh.model.geo.addCircleArc(7, 1, 12, 11)
    gmsh.model.geo.addCircleArc(8, 3, 13, 12)

    gmsh.model.geo.addCurveLoop([1, 10, -5, -9], 13)
    gmsh.model.geo.addSurfaceFilling([13], 1)
    gmsh.model.geo.addCurveLoop([2, 11, -6, -10], 14)
    gmsh.model.geo.addSurfaceFilling([14], 2)
    gmsh.model.geo.addCurveLoop([3, 12, -7, -11], 15)
    gmsh.model.geo.addSurfaceFilling([15], 3)
    gmsh.model.geo.addCurveLoop([4, 9, -8, -12], 16)
    gmsh.model.geo.addSurfaceFilling([16], 4)

    sf = gmsh.model.geo.copy([(2, 1), (2, 2), (2, 3), (2, 4)])
    gmsh.model.geo.rotate(sf, 0, 0, 0, 0, 0, 1, -pi/2)
    sf = gmsh.model.geo.copy([(2, 1), (2, 2), (2, 3), (2, 4)])
    gmsh.model.geo.rotate(sf, 0, 0, 0, 0, 0, 1, pi/2)
    sf = gmsh.model.geo.copy([(2, 1), (2, 2), (2, 3), (2, 4)])
    gmsh.model.geo.rotate(sf, 0, 0, 0, 0, 0, 1, pi)

    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.MshFileVersion",2)
    gmsh.model.mesh.generate(2)
    gmsh.fltk.run()
    gmsh.write(fno)
    gmsh.finalize()

    m = CompScienceMeshes.read_gmsh_mesh(fno)
    rm(fno)
    return m
end

meshtorus(;innerradius, outerradius, h) = meshtorus(innerradius, outerradius, h)



"""
    meshsquaretorus(width, height, holewidth, h)
    meshsquaretorus(;width, height, holewidth, h)

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
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 101)
    
    gmsh.model.geo.addPoint(width/2, -width/2, -height/2, h, 5)
    gmsh.model.geo.addPoint(width/2, width/2, -height/2, h, 6)
    gmsh.model.geo.addPoint(-width/2, width/2, -height/2, h, 7)
    gmsh.model.geo.addPoint(-width/2, -width/2, -height/2, h, 8)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 5, 8)

    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 102)
    gmsh.model.geo.addPlaneSurface([101, 102], 1)
    gmsh.model.geo.extrude([(2, 1)], 0, 0, height)

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

meshsquaretorus(;width, height, holewidth, h) = meshsquaretorus(width, height, holewidth, h)

"""
    meshsquaretorus4(width, height, holewidth, h)
    meshsquaretorus4(;width, height, holewidth, h)

Create a mesh of a square torus of size `width` and `height` with 4 holes of size `holewidth` and `height`

The target edge size is `h`.
"""
function meshsquaretorus4(width, height, holewidth, h)
    @assert (2*holewidth) < width
    
    fno = tempname() * ".msh"
    gmsh.initialize()
    gmsh.model.add("squaretorus4")

    # bottom plate
    gmsh.model.geo.addPoint(width/2, -width/2, -height/2, h, 1)
    gmsh.model.geo.addPoint(width/2, width/2, -height/2, h, 2)
    gmsh.model.geo.addPoint(-width/2, width/2, -height/2, h, 3)
    gmsh.model.geo.addPoint(-width/2, -width/2, -height/2, h, 4)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 101)

    gmsh.model.geo.addPoint(holewidth/2 + width/4, -holewidth/2 - width/4, -height/2, h, 5)
    gmsh.model.geo.addPoint(holewidth/2 + width/4, holewidth/2 - width/4, -height/2, h, 6)
    gmsh.model.geo.addPoint(-holewidth/2 + width/4, holewidth/2 - width/4, -height/2, h, 7)
    gmsh.model.geo.addPoint(-holewidth/2 + width/4, -holewidth/2 - width/4, -height/2, h, 8)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 5, 8)

    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 102)

    hole2 = gmsh.model.geo.copy([(1, 5), (1, 6), (1, 7), (1, 8)])
    gmsh.model.geo.translate(hole2, 0, width/2, 0)
    curveloop2 = [tag for (dim, tag) in hole2]
    gmsh.model.geo.addCurveLoop(curveloop2, 103)

    hole3 = gmsh.model.geo.copy([(1, 5), (1, 6), (1, 7), (1, 8)])
    gmsh.model.geo.translate(hole3, -width/2, width/2, 0)
    curveloop3 = [tag for (dim, tag) in hole3]
    gmsh.model.geo.addCurveLoop(curveloop3, 104)

    hole4 = gmsh.model.geo.copy([(1, 5), (1, 6), (1, 7), (1, 8)])
    gmsh.model.geo.translate(hole4, -width/2, 0, 0)
    curveloop4 = [tag for (dim, tag) in hole4]
    gmsh.model.geo.addCurveLoop(curveloop4, 105)

    gmsh.model.geo.addPlaneSurface([101, 102, 103, 104, 105], 1)
    gmsh.model.geo.extrude([(2, 1)], 0, 0, height)

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

meshsquaretorus4(;width, height, holewidth, h) = meshsquaretorus4(width, height, holewidth, h)