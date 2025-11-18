import panda_py
import frankx
import spatialmath as sm
import numpy as np
from roboticstoolbox.models import Panda

panda = Panda()

for i in range(4):
    q = panda.random_q()
    T_0p = sm.SE3(panda_py.fk(q), check=False)
    T_0f = sm.SE3(np.array(frankx.Kinematics.forward(q)).reshape(4, 4, order="F"))

    T_pf = T_0p.inv() * T_0f
    T_fp = T_0f.inv() * T_0p
    print(T_pf)
    print(T_fp)
