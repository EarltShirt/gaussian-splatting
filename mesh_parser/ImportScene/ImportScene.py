"""
   Copyright (C) 2017 Autodesk, Inc.
   All rights reserved.

   Use of this software is subject to the terms of the Autodesk license agreement
   provided at the time of installation or download, or which otherwise accompanies
   this software in either electronic or hard copy form.
 
"""

import os, sys

from DisplayGlobalSettings  import *
from DisplayHierarchy       import DisplayHierarchy
import numpy as np
import json
# from DisplayMarker          import DisplayMarker
# from DisplayMesh            import DisplayMesh
# from DisplayUserProperties  import DisplayUserProperties
# from DisplayPivotsAndLimits import DisplayPivotsAndLimits
# from DisplaySkeleton        import DisplaySkeleton
# from DisplayNurb            import DisplayNurb
# from DisplayPatch           import DisplayPatch
# from DisplayCamera          import DisplayCamera
# from DisplayLight           import DisplayLight
# from DisplayLodGroup        import DisplayLodGroup
# from DisplayPose            import DisplayPose
# from DisplayAnimation       import DisplayAnimation
# from DisplayGenericInfo     import DisplayGenericInfo


def DisplayHierarchy(pScene):
    names = []
    lRootNode = pScene.GetRootNode()

    for i in range(lRootNode.GetChildCount()):
        DisplayNodeHierarchy(lRootNode.GetChild(i), 0, names)
    
    return names

def DisplayNodeHierarchy(pNode, pDepth, names):
    lString = ""
    for i in range(pDepth):
        lString += "     "

    lString += pNode.GetName()
    names.append(pNode.GetName())

    print(lString)

    for i in range(pNode.GetChildCount()):
        DisplayNodeHierarchy(pNode.GetChild(i), pDepth + 1, names)

def get_BBox_list(names):
    BBBounds = {}
    for name in names:
        BBox = get_BBox(name)
        BBBounds[name] = {'min': BBox[0], 'max': BBox[1]}
    return BBBounds

def get_BBox(name):
    lNode = lScene.FindNodeByName(name)
    mesh = lNode.GetMesh()
    print(f'Node : {name}')
    mesh.ComputeBBox()
    lmin = mesh.BBoxMin.Get()
    lmax = mesh.BBoxMax.Get()
    Display3DVector("   Min Bounds : (" , lmin,")")
    Display3DVector("   Max Bounds : (" , lmax,")")
    min = [lmin[0], lmin[1], lmin[2]]
    max = [lmax[0], lmax[1], lmax[2]]
    return [min, max]

if __name__ == "__main__":
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        from FbxCommon import *
    except ImportError:
        print("Error: module FbxCommon failed to import.\n")
        sys.exit(1)

    # Prepare the FBX SDK.
    lSdkManager, lScene = InitializeSdkObjects()
    # Load the scene.

    # The example can take a FBX file as an argument.
    if len(sys.argv) > 1:
        print("\n\nFile: %s\n" % sys.argv[1])
        lResult = LoadScene(lSdkManager, lScene, sys.argv[1])
    else :
        lResult = False

        print("\n\nUsage: ImportScene <FBX file name>\n")

    print("\n\n---------\nHierarchy\n---------\n")
    names = DisplayHierarchy(lScene)[4:]

    print(names)

    print("Bounding Box Retrieval")
    bounds = get_BBox_list(names)
    # print(bounds)

    print(f'Min Boundings of {names[3]} : {bounds[names[3]]["min"]}')

    # Save bounding box data to a json file (should be changed to the data repository once one is created)
    with open('../bounding_boxes.json', 'w') as f:
        json.dump(bounds, f)


    # Destroy all objects created by the FBX SDK.
    lSdkManager.Destroy()
   
    print("LOADING TEST :")
    with open('../bounding_boxes.json', 'r') as f:
        bounds_test = json.load(f)

    part1 = []
    part2 = []
    part3 = []
    part4 = []
    for key in bounds_test.keys():
        # verify if the string 'part1' is present in the key's name
        if 'part1' in key:
            print("yes : ", key)
            part1.append(key)
        elif 'part2' in key:
            part2.append(key)
        elif 'part3' in key:
            part3.append(key)
        elif 'part4' in key:
            part4.append(key)
    print(part1)
    print(part2)
    print(part3)
    print(part4)
    # for name in bounds_test:
    #     print(f'Node : {name}')
    #     print(f'Min Bounds : {bounds_test[name]["min"]}')
    #     print(f'Max Bounds : {bounds_test[name]["max"]}')

    if lResult:
        sys.exit(0)
    else:
        sys.exit(-1)
