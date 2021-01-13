import unittest
import numpy as np
import operator
from wavevector import *
from wavefunction import *

class Wavevector_Tests(unittest.TestCase):
    def test_init(self):
        """ wavevector initialization """
        x = np.asarray([1. + 0.j,2,3])
        wv1 = Wavevector(x)
        self.assertEqual(str(wv1), '[1.+0.j 2.+0.j 3.+0.j]')

    def test_algebra(self):
        """ test arithmetic operations between two wvecs, and between wvec and constant """
        wv1 = Wavevector([1,2,3])
        self.assertEqual(str(wv1 + wv1), '[2.+0.j 4.+0.j 6.+0.j]')
        self.assertEqual(str(3 + wv1), '[4.+0.j 5.+0.j 6.+0.j]')
        self.assertEqual(str(wv1 *  wv1), '[1.+0.j 4.+0.j 9.+0.j]')
        self.assertEqual(str(3*wv1), '[3.+0.j 6.+0.j 9.+0.j]')

    def test_slice(self):
        """ test numpy slicing of wvec """
        wv1 = Wavevector([1,2,3])
        self.assertEqual(str(wv1[1:]), '[2.+0.j 3.+0.j]')
        self.assertEqual(type(wv1[1:]), Wavevector)
        test_arr = np.asarray([1,2,3]).astype(complex)
        self.assertEqual(str(test_arr.view(Wavevector)), '[1.+0.j 2.+0.j 3.+0.j]')
        self.assertEqual(type(test_arr.view(Wavevector)), Wavevector)

    def test_from_wf(self):
        cls_type = Wavefunction
        wf1 = cls_type.init_gaussian((0, 1))
        wv1 = Wavevector.from_wavefunction(wf1,(-10, 10, 21))
        self.assertEqual(wv1[10], 1/(2*π)**0.25+0j)

if __name__ == '__main__':
    unittest.main()
