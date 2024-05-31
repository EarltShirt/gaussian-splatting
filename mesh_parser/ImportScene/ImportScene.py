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
    
    return names[3:]

def DisplayNodeHierarchy(pNode, pDepth, names):
    lString = ""
    for i in range(pDepth):
        lString += "     "

    lString += pNode.GetName()
    names.append(pNode.GetName())

    print(lString)

    for i in range(pNode.GetChildCount()):
        DisplayNodeHierarchy(pNode.GetChild(i), pDepth + 1, names)



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
    names = DisplayHierarchy(lScene)

    print(names)


    # Destroy all objects created by the FBX SDK.
    lSdkManager.Destroy()
   
    if lResult:
        sys.exit(0)
    else:
        sys.exit(-1)
