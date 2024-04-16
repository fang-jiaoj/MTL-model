# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Node and edge featurization for molecular graphs.
# pylint: disable= no-member, arguments-differ, invalid-name

import itertools
import os.path as osp

from collections import defaultdict
from functools import partial
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

import numpy as np
import torch
import dgl.backend as F

__all__ = ['one_hot_encoding',
           'atom_type_one_hot',
           'atomic_number_one_hot',
           'atomic_number',
           'atom_degree_one_hot',
           'atom_degree',
           'atom_total_degree_one_hot',
           'atom_total_degree',
           'atom_explicit_valence_one_hot',
           'atom_explicit_valence',
           'atom_implicit_valence_one_hot',
           'atom_implicit_valence',
           'atom_hybridization_one_hot',
           'atom_total_num_H_one_hot',
           'atom_total_num_H',
           'atom_formal_charge_one_hot',
           'atom_formal_charge',
           'atom_num_radical_electrons_one_hot',
           'atom_num_radical_electrons',
           'atom_is_aromatic_one_hot',
           'atom_is_aromatic',
           'atom_is_in_ring_one_hot',
           'atom_is_in_ring',
           'atom_chiral_tag_one_hot',
           'atom_chirality_type_one_hot',
           'atom_mass',
           'atom_is_chiral_center',
           'ConcatFeaturizer',
           'BaseAtomFeaturizer',
           'CanonicalAtomFeaturizer',
           'WeaveAtomFeaturizer',
           'PretrainAtomFeaturizer',
           'AttentiveFPAtomFeaturizer',
           'PAGTNAtomFeaturizer',
           'bond_type_one_hot',
           'bond_is_conjugated_one_hot',
           'bond_is_conjugated',
           'bond_is_in_ring_one_hot',
           'bond_is_in_ring',
           'bond_stereo_one_hot',
           'bond_direction_one_hot',
           'BaseBondFeaturizer',
           'CanonicalBondFeaturizer',
           'WeaveEdgeFeaturizer',
           'PretrainBondFeaturizer',
           'AttentiveFPBondFeaturizer',
           'PAGTNEdgeFeaturizer']

def one_hot_encoding(x,allowable_set,encode_unknown=False):
    """定义一个独热编码对特征取值进行编码"""
    """One-hot encoding.

        Parameters
        ----------
        x
            Value to encode.
        allowable_set : list
            The elements of the allowable_set should be of the
            same type as x.
        encode_unknown : bool
            If True, map inputs not in the allowable set to the
            additional last element.

        Returns
        -------
        list
            List of boolean values where at most one value is True.
            The list is of length ``len(allowable_set)`` if ``encode_unknown=False``
            and ``len(allowable_set) + 1`` otherwise.

        Examples
        --------
        >>> from dgllife.utils import one_hot_encoding
        >>> one_hot_encoding('C', ['C', 'O'])
        [True, False]
        >>> one_hot_encoding('S', ['C', 'O'])
        [False, False]
        >>> one_hot_encoding('S', ['C', 'O'], encode_unknown=True)
        [False, False, True]
        """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)
    if encode_unknown and (x not in allowable_set):
        x = None
    return list(map(lambda s:x==s ,allowable_set))

#################################################################
# Atom featurization
#################################################################

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """对原子类型进行独热编码"""
    """One hot encoding for the type of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Atom types to consider. Default: ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``,
        ``Cl``, ``Br``, ``Mg``, ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``,
        ``K``, ``Tl``, ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
        ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``, ``Cr``,
        ``Pt``, ``Hg``, ``Pb``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atomic_number
    atomic_number_one_hot
    """
    if allowable_set is None:
        ##包含一组预定义元素符号的列表
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        ###问题：原子的定义依据是什么？可不可以换？
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown) ##原子符号的独热编码结果

def atomic_number_one_hot(atom, allowable_set=None, encode_unknown=False):
    ###对原子序数进行独热编码，默认使用从1到100的整数序列
    """One hot encoding for the atomic number of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atomic numbers to consider. Default: ``1`` - ``100``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atomic_number
    atom_type_one_hot
    """
    if allowable_set is None:
        allowable_set = list(range(1, 101))
    return one_hot_encoding(atom.GetAtomicNum(), allowable_set, encode_unknown) ###原子原子序数的独热编码结果

def atomic_number(atom):
    """获取 RDKit 中的原子的原子序数"""
    """Get the atomic number for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
       List containing one int only.

    See Also
    --------
    atomic_number_one_hot
    atom_type_one_hot
    """
    ###该函数返回一个包含一个整数的列表，即原子的原子序数
    return [atom.GetAtomicNum()]

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """函数是用于对原子的度数进行独热编码的"""
    """One hot encoding for the degree of an atom.

    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom degrees to consider. Default: ``0`` - ``10``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_degree
    atom_total_degree
    atom_total_degree_one_hot
    """
    if allowable_set is None:
        allowable_set = list(range(11)) ##0 到 10 的整数列表
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown) ##返回对原子度数的独热编码

def atom_degree(atom):
    """atom_degree 函数是用来获取原子的度数的函数"""
    """Get the degree of an atom.

    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_degree_one_hot
    atom_total_degree
    atom_total_degree_one_hot
    """
    ##返回一个包含一个整数的列表，该整数表示该原子的度
    return [atom.GetDegree()]

def atom_total_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """用于对原子的总度数进行独热编码的函数,
    原子的度（Degree）： 仅考虑直接连接的边的数量，不考虑隐式氢,
    原子的总度数（Total Degree）： 包括与该原子直接相连的边的数量以及该原子的隐式氢的数量"""
    """One hot encoding for the degree of an atom including Hs.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list
        Total degrees to consider. Default: ``0`` - ``5``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    See Also
    --------
    one_hot_encoding
    atom_degree
    atom_degree_one_hot
    atom_total_degree
    """
    if allowable_set is None:
        allowable_set = list(range(6)) ##总度数范围，默认为 0 到 5
        ###返回原子总度的独热编码
    return one_hot_encoding(atom.GetTotalDegree(), allowable_set, encode_unknown)

def atom_total_degree(atom):
    """计算该原子的总度，就是包括所连氢的数量"""
    """The degree of an atom including Hs.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_total_degree_one_hot
    atom_degree
    atom_degree_one_hot
    """
    return [atom.GetTotalDegree()]

def atom_explicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """对原子的显式化合价进行独热编码"""
    """One hot encoding for the explicit valence of an aotm.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom explicit valences to consider. Default: ``1`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_explicit_valence
    """
    if allowable_set is None:
        allowable_set = list(range(1, 7)) ##可接受的原子显式化合价值的列表。默认为 1 到 6
        ##原子的显式化合价的独热编码
    return one_hot_encoding(atom.GetExplicitValence(), allowable_set, encode_unknown)

def atom_explicit_valence(atom):
    """得到原子显式化合价"""
    """Get the explicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_explicit_valence_one_hot
    """
    return [atom.GetExplicitValence()]

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """用于对原子的隐式化合价进行独热编码，
    显式化合价通常通过分子结构图中的共价键和非共价电子对来确定，
    隐式化合价考虑了原子周围的邻接原子，以及这些邻接原子通过共价键共享的电子"""
    """One hot encoding for the implicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom implicit valences to consider. Default: ``0`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    atom_implicit_valence
    """
    if allowable_set is None:
        allowable_set = list(range(7)) ##原子隐式化合价值的列表。默认为 0 到 6
        ###返回原子的隐式化合价的独热编码
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)

def atom_implicit_valence(atom):
    """计算出原子的隐式化合价"""
    """Get the implicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Reurns
    ------
    list
        List containing one int only.

    See Also
    --------
    atom_implicit_valence_one_hot
    """
    return [atom.GetImplicitValence()]

# pylint: disable=I1101
def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """用于对原子的杂化状态（hybridization）进行独热编码"""
    """One hot encoding for the hybridization of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.HybridizationType
        Atom hybridizations to consider. Default: ``Chem.rdchem.HybridizationType.SP``,
        ``Chem.rdchem.HybridizationType.SP2``, ``Chem.rdchem.HybridizationType.SP3``,
        ``Chem.rdchem.HybridizationType.SP3D``, ``Chem.rdchem.HybridizationType.SP3D2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        ###原子杂化状态的列表，默认为SP，SP2，SP3，SP3D，SP3D2
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)

def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    """用于对原子的总氢数进行独热编码"""
    """One hot encoding for the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Total number of Hs to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_total_num_H
    """
    if allowable_set is None:
        allowable_set = list(range(5))  ##要考虑的原子总氢数的列表，默认为0到4
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)

def atom_total_num_H(atom):
    """计算出原子的总氢数"""
    """Get the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_total_num_H_one_hot
    """
    return [atom.GetTotalNumHs()]

def atom_formal_charge_one_hot(atom, allowable_set=None, encode_unknown=False):
    """这个函数用于对原子的形式电荷进行独热编码"""
    """One hot encoding for the formal charge of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Formal charges to consider. Default: ``-2`` - ``2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_formal_charge
    """
    if allowable_set is None:
        allowable_set = list(range(-2, 3))
    return one_hot_encoding(atom.GetFormalCharge(), allowable_set, encode_unknown)

def atom_formal_charge(atom):
    """计算出原子的形式电荷"""
    """Get formal charge for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_formal_charge_one_hot
    """
    return [atom.GetFormalCharge()]

def atom_partial_charge(atom):
    """这个函数用于获取原子的Gasteiger偏电荷"""
    """Get Gasteiger partial charge for an atom.

    For using this function, you must have called ``AllChem.ComputeGasteigerCharges(mol)``
    to compute Gasteiger charges.

    Occasionally, we can get nan or infinity Gasteiger charges, in which case we will set
    the result to be 0.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one float only.
    """
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0
        ###返回原子的偏电荷
    return [float(gasteiger_charge)]

def atom_num_radical_electrons_one_hot(atom, allowable_set=None, encode_unknown=False):
    """用于对原子的自由基电子数进行一热编码"""
    """One hot encoding for the number of radical electrons of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Number of radical electrons to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_num_radical_electrons
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetNumRadicalElectrons(), allowable_set, encode_unknown)

def atom_num_radical_electrons(atom):
    """计算出原子的自由基电子数"""
    """Get the number of radical electrons for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_num_radical_electrons_one_hot
    """
    return [atom.GetNumRadicalElectrons()]

def atom_is_aromatic_one_hot(atom, allowable_set=None, encode_unknown=False):
    """原子是否是芳香的进行一热编码"""
    """One hot encoding for whether the atom is aromatic.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_is_aromatic
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.GetIsAromatic(), allowable_set, encode_unknown)

def atom_is_aromatic(atom):
    """计算出原子是否具有芳香性"""
    """Get whether the atom is aromatic.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    atom_is_aromatic_one_hot
    """
    return [atom.GetIsAromatic()]

def atom_is_in_ring_one_hot(atom, allowable_set=None, encode_unknown=False):
    """这个函数用于对原子是否位于环中进行一热编码"""
    """One hot encoding for whether the atom is in ring.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_is_in_ring
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.IsInRing(), allowable_set, encode_unknown)

def atom_is_in_ring(atom):
    """计算出原子是否处于环中"""
    """Get whether the atom is in ring.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    atom_is_in_ring_one_hot
    """
    return [atom.IsInRing()]

def atom_chiral_tag_one_hot(atom, allowable_set=None, encode_unknown=False):
    """这个函数用于对原子的手性标记进行一热编码"""
    """One hot encoding for the chiral tag of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.ChiralType
        Chiral tags to consider. Default: ``rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_OTHER``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    one_hot_encoding
    atom_chirality_type_one_hot
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                         Chem.rdchem.ChiralType.CHI_OTHER]
        ##返回手性标记的独热编码
    return one_hot_encoding(atom.GetChiralTag(), allowable_set, encode_unknown)

def atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """这个函数用于对原子的手性类型进行独热编码"""
    """One hot encoding for the chirality type of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Chirality types to consider. Default: ``R``, ``S``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    one_hot_encoding
    atom_chiral_tag_one_hot
    """
    if not atom.HasProp('_CIPCode'): ##如果原子没有CIP编码，函数返回[False, False]，表示未知手性
        return [False, False]

    if allowable_set is None:
        allowable_set = ['R', 'S']
        ###返回一个包含一个布尔值的列表，表示原子的手性类型
    return one_hot_encoding(atom.GetProp('_CIPCode'), allowable_set, encode_unknown)

def atom_mass(atom, coef=0.01):
    """这个函数用于获取原子的质量并进行缩放"""
    """Get the mass of an atom and scale it.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    coef : float
        The mass will be multiplied by ``coef``.

    Returns
    -------
    list
        List containing one float only.
    """
    return [atom.GetMass() * coef]

def atom_is_chiral_center(atom):
    """这个函数用于获取原子是否是手性中心"""
    """Get whether the atom is chiral center

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.
    """
    return [atom.HasProp('_ChiralityPossible')]

class ConcatFeaturizer(object):
    """Concatenate the evaluation results of multiple functions as a single feature.

    Parameters
    ----------
    func_list : list
        List of functions for computing molecular descriptors from objects of a same
        particular data type, e.g. ``rdkit.Chem.rdchem.Atom``. Each function is of signature
        ``func(data_type) -> list of float or bool or int``. The resulting order of
        the features will follow that of the functions in the list.

    Examples
    --------

    Setup for demo.

    >>> from dgllife.utils import ConcatFeaturizer
    >>> from rdkit import Chem
    >>> smi = 'CCO'
    >>> mol = Chem.MolFromSmiles(smi)

    Concatenate multiple atom descriptors as a single node feature.

    >>> from dgllife.utils import atom_degree, atomic_number, BaseAtomFeaturizer
    >>> # Construct a featurizer for featurizing one atom a time
    >>> atom_concat_featurizer = ConcatFeaturizer([atom_degree, atomic_number])
    >>> # Construct a featurizer for featurizing all atoms in a molecule
    >>> mol_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})
    >>> mol_atom_featurizer(mol)
    {'h': tensor([[1., 6.],
                  [2., 6.],
                  [1., 8.]])}

    Conctenate multiple bond descriptors as a single edge feature.

    >>> from dgllife.utils import bond_type_one_hot, bond_is_in_ring, BaseBondFeaturizer
    >>> # Construct a featurizer for featurizing one bond a time
    >>> bond_concat_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])
    >>> # Construct a featurizer for featurizing all bonds in a molecule
    >>> mol_bond_featurizer = BaseBondFeaturizer({'h': bond_concat_featurizer})
    >>> mol_bond_featurizer(mol)
    {'h': tensor([[1., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0.]])}
    """
    def __init__(self, func_list):
        ###接受一个参数 func_list，该参数是一个包含多个函数的列表,这些函数用以计算描述符
        self.func_list = func_list

    def __call__(self, x):
        """Featurize the input data.

        Parameters
        ----------
        x :
            Data to featurize.

        Returns
        -------
        list
            List of feature values, which can be of type bool, float or int.
        """
        ###返回特征列表
        return list(itertools.chain.from_iterable(
            [func(x) for func in self.func_list]))

class BaseAtomFeaturizer(object):
    """An abstract class for atom featurizers.

    Loop over all atoms in a molecule and featurize them with the ``featurizer_funcs``.

    **We assume the resulting DGLGraph will not contain any virtual nodes and a node i in the
    graph corresponds to exactly atom i in the molecule.**

    Parameters
    ----------
    featurizer_funcs : dict
        Mapping feature name to the featurization function.
        Each function is of signature ``func(rdkit.Chem.rdchem.Atom) -> list or 1D numpy array``.
    feat_sizes : dict
        Mapping feature name to the size of the corresponding feature. If None, they will be
        computed when needed. Default: None.

    Examples
    --------

    >>> from dgllife.utils import BaseAtomFeaturizer, atom_mass, atom_degree_one_hot
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = BaseAtomFeaturizer({'mass': atom_mass, 'degree': atom_degree_one_hot})
    >>> atom_featurizer(mol)
    {'mass': tensor([[0.1201],
                     [0.1201],
                     [0.1600]]),
     'degree': tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size for atom mass
    >>> print(atom_featurizer.feat_size('mass'))
    1
    >>> # Get feature size for atom degree
    >>> print(atom_featurizer.feat_size('degree'))
    11

    See Also
    --------
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    PretrainAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        ####featurizer_funcs 是一个包含特征化函数的字典，其中键是特征的名称，值是用于特征化的函数
        if feat_sizes is None:
            feat_sizes = dict()  #feat_sizes，则默认为一个空字典
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        """获取指定特征（或唯一的特征，如果没有指定特征名称）feat_name的大小"""
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0] ###当只有一个特征时，直接获取特征名2

        if feat_name not in self.featurizer_funcs:
            ###如果 feat_name 不在 featurizer_funcs 中，会引发 ValueError，指示提供的 feat_name 无效
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        if feat_name not in self._feat_sizes:
            atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
            #####应用特定特征提取函数，然后存储特征的大小
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))

        return self._feat_sizes[feat_name] ###获取指定特征（或唯一的特征，如果没有指定特征名称）的大小

    def __call__(self, mol):
        """Featurize all atoms in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_atoms = mol.GetNumAtoms()  #首先获取其包含的原子数量 num_atoms
        atom_features = defaultdict(list)  #用于存储每个原子的特征

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))  ###获得{特征名：[原子的特征值]}

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list) ###将其特征值列表堆叠（stack）成一个 numpy 数组
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))
            ###将所有特征化后的结果存储在一个字典 processed_features 中，其中键是特征名称，值是对应的张量

        return processed_features  ##将每个原子的多个特征组合成一个字典，其中包含所有的特征名称和对应的张量

class CanonicalAtomFeaturizer(BaseAtomFeaturizer):
    """A default featurizer for atoms.

    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``, ``Cl``, ``Br``, ``Mg``,
      ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``, ``K``, ``Tl``,
      ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
      ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``,
      ``Cr``, ``Pt``, ``Hg``, ``Pb``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 10``.
    * **One hot encoding of the number of implicit Hs on the atom**. The supported
      possibilities include ``0 - 6``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import CanonicalAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                      1., 0.],
                     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
                      0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
                      0., 0.]])}

    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size('feat'))
    74

    See Also
    --------
    BaseAtomFeaturizer
    WeaveAtomFeaturizer
    PretrainAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        ###键是 atom_data_field（默认为 'h'），值是一个 ConcatFeaturizer 类的实例
        ###一个列表，包含了多个特征提取函数
        super(CanonicalAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [atom_type_one_hot,
                 atom_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge,
                 atom_is_aromatic,
                 atom_num_radical_electrons,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_total_num_H_one_hot,
                 atom_is_chiral_center,
                 atom_chirality_type_one_hot]
            )})

class WeaveAtomFeaturizer(object):
    """Atom featurizer in Weave.

    The atom featurization performed in `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__, which considers:

    * atom types
    * chirality
    * formal charge
    * partial charge
    * aromatic atom
    * hybridization
    * hydrogen bond donor
    * hydrogen bond acceptor
    * the number of rings the atom belongs to for ring size between 3 and 8

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.
    atom_types : list of str or None
        Atom types to consider for one-hot encoding. If None, we will use a default
        choice of ``'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'``.
    chiral_types : list of Chem.rdchem.ChiralType or None
        Atom chirality to consider for one-hot encoding. If None, we will use a default
        choice of ``Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``.
    hybridization_types : list of Chem.rdchem.HybridizationType or None
        Atom hybridization types to consider for one-hot encoding. If None, we will use a
        default choice of ``Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3``.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import WeaveAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = WeaveAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0418,  0.0000,
                       0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000],
                     [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0402,  0.0000,
                       0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000],
                     [ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.3967,  0.0000,
                       0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000]])}
    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size())
    27

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    PretrainAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h', atom_types=None, chiral_types=None,
                 hybridization_types=None):
        super(WeaveAtomFeaturizer, self).__init__()

        self._atom_data_field = atom_data_field

        if atom_types is None:
            atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        self._atom_types = atom_types

        if chiral_types is None:
            chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
        self._chiral_types = chiral_types

        if hybridization_types is None:
            hybridization_types = [Chem.rdchem.HybridizationType.SP,
                                   Chem.rdchem.HybridizationType.SP2,
                                   Chem.rdchem.HybridizationType.SP3]
        self._hybridization_types = hybridization_types

        self._featurizer = ConcatFeaturizer([
            partial(atom_type_one_hot, allowable_set=atom_types, encode_unknown=True),
            partial(atom_chiral_tag_one_hot, allowable_set=chiral_types),
            atom_formal_charge, atom_partial_charge, atom_is_aromatic,
            partial(atom_hybridization_one_hot, allowable_set=hybridization_types)
        ])

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._atom_data_field]

        return feats.shape[-1]

    def get_donor_acceptor_info(self, mol_feats):
        """Bookkeep whether an atom is donor/acceptor for hydrogen bonds.

        Parameters
        ----------
        mol_feats : tuple of rdkit.Chem.rdMolChemicalFeatures.MolChemicalFeature
            Features for molecules.

        Returns
        -------
        is_donor : dict
            Mapping atom ids to binary values indicating whether atoms
            are donors for hydrogen bonds
        is_acceptor : dict
            Mapping atom ids to binary values indicating whether atoms
            are acceptors for hydrogen bonds
        """
        is_donor = defaultdict(bool)
        is_acceptor = defaultdict(bool)
        # Get hydrogen bond donor/acceptor information
        for feats in mol_feats:
            if feats.GetFamily() == 'Donor':
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_donor[u] = True
            elif feats.GetFamily() == 'Acceptor':
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_acceptor[u] = True

        return is_donor, is_acceptor

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping atom_data_field as specified in the input argument to the atom
            features, which is a float32 tensor of shape (N, M), N is the number of
            atoms and M is the feature size.
        """
        atom_features = []

        AllChem.ComputeGasteigerCharges(mol)
        num_atoms = mol.GetNumAtoms()

        # Get information for donor and acceptor
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        is_donor, is_acceptor = self.get_donor_acceptor_info(mol_feats)

        # Get a symmetrized smallest set of smallest rings
        # Following the practice from Chainer Chemistry (https://github.com/chainer/
        # chainer-chemistry/blob/da2507b38f903a8ee333e487d422ba6dcec49b05/chainer_chemistry/
        # dataset/preprocessors/weavenet_preprocessor.py)
        sssr = Chem.GetSymmSSSR(mol)

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            # Features that can be computed directly from RDKit atom instances, which is a list
            feats = self._featurizer(atom)
            # Donor/acceptor indicator
            feats.append(float(is_donor[i]))
            feats.append(float(is_acceptor[i]))
            # Count the number of rings the atom belongs to for ring size between 3 and 8
            count = [0 for _ in range(3, 9)]
            for ring in sssr:
                ring_size = len(ring)
                if i in ring and 3 <= ring_size <= 8:
                    count[ring_size - 3] += 1
            feats.extend(count)
            atom_features.append(feats)
        atom_features = np.stack(atom_features)

        return {self._atom_data_field: F.zerocopy_from_numpy(atom_features.astype(np.float32))}

class PretrainAtomFeaturizer(object):
    """AtomFeaturizer in Strategies for Pre-training Graph Neural Networks.

    The atom featurization performed in `Strategies for Pre-training Graph Neural Networks
    <https://arxiv.org/abs/1905.12265>`__, which considers:

    * atomic number
    * chirality

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atomic_number_types : list of int or None
        Atomic number types to consider for one-hot encoding. If None, we will use a default
        choice of 1-118.
    chiral_types : list of Chem.rdchem.ChiralType or None
        Atom chirality to consider for one-hot encoding. If None, we will use a default
        choice of ``Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER``.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import PretrainAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = PretrainAtomFeaturizer()
    >>> atom_featurizer(mol)
    {'atomic_number': tensor([5, 5, 7]), 'chirality_type': tensor([0, 0, 0])}

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atomic_number_types=None, chiral_types=None):
        if atomic_number_types is None:
            atomic_number_types = list(range(1, 119))
        self._atomic_number_types = atomic_number_types

        if chiral_types is None:
            chiral_types = [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ]
        self._chiral_types = chiral_types

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping 'atomic_number' and 'chirality_type' to separately an int64 tensor
            of shape (N, 1), N is the number of atoms
        """
        atom_features = []
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_features.append([
                self._atomic_number_types.index(atom.GetAtomicNum()),
                self._chiral_types.index(atom.GetChiralTag())
            ])
        atom_features = np.stack(atom_features)
        atom_features = F.zerocopy_from_numpy(atom_features.astype(np.int64))

        return {
            'atomic_number': atom_features[:, 0],
            'chirality_type': atom_features[:, 1]
        }

class AttentiveFPAtomFeaturizer(BaseAtomFeaturizer):
    """用于在AttentiveFP中对原子进行特征化"""
    """The atom featurizer used in AttentiveFP

    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``B``, ``C``, ``N``, ``O``, ``F``, ``Si``, ``P``, ``S``, ``Cl``, ``As``,
      ``Se``, ``Br``, ``Te``, ``I``, ``At``, and ``other``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 5``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``, and ``other``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.
    * **Whether the atom is chiral center**
    * **One hot encoding of the atom chirality type**. The supported possibilities include
      ``R``, and ``S``.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import AttentiveFPAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                      0., 0., 0.],
                     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                      0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                      0., 0., 0.]])}

    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size('feat'))
    39

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    PretrainAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        ######键是 atom_data_field（默认为 'h'），值是一个 ConcatFeaturizer 类的实例
        ###一个列表，包含了多个特征提取函数
        super(AttentiveFPAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [partial(atom_type_one_hot, allowable_set=[
                    'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S',
                    'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'], encode_unknown=True),
                 partial(atom_degree_one_hot, allowable_set=list(range(6))),
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 partial(atom_hybridization_one_hot, encode_unknown=True),
                 atom_is_aromatic,
                 atom_total_num_H_one_hot,
                 atom_is_chiral_center,
                 atom_chirality_type_one_hot]
            )})

class PAGTNAtomFeaturizer(BaseAtomFeaturizer):
    """The atom featurizer used in PAGTN

    PAGTN is introduced in
    `Path-Augmented Graph Transformer Network. <https://arxiv.org/abs/1905.12712>`__

    The atom features include:

    * **One hot encoding of the atom type**.
    * **One hot encoding of formal charge of the atom**.
    * **One hot encoding of the atom degree**
    * **One hot encoding of explicit valence of an atom**. The supported possibilities
      include ``0 - 6``.
    * **One hot encoding of implicit valence of an atom**. The supported possibilities
      include ``0 - 5``.
    * **Whether the atom is aromatic**.

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import PAGTNAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('C')
    >>> atom_featurizer = PAGTNAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 1., 0., 0.]])}
    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size())
    94

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    PretrainAtomFeaturizer
    WeaveAtomFeaturizer
    AttentiveFPAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                   'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                   'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                   'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                   'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re',
                   'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm',
                   'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', '*', 'UNK']

        super(PAGTNAtomFeaturizer, self).__init__(
            featurizer_funcs={
                atom_data_field: ConcatFeaturizer([partial(atom_type_one_hot,
                                                           allowable_set=SYMBOLS,
                                                           encode_unknown=False),
                                                   atom_formal_charge_one_hot,
                                                   atom_degree_one_hot,
                                                   partial(atom_explicit_valence_one_hot,
                                                           allowable_set=list(range(7)),
                                                           encode_unknown=False),
                                                   partial(atom_implicit_valence_one_hot,
                                                           allowable_set=list(range(6)),
                                                           encode_unknown=False),
                                                   atom_is_aromatic])})

class WeaveAtomFeaturizer(object):
    """Atom featurizer in Weave.

    The atom featurization performed in `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__, which considers:

    * atom types
    * chirality
    * formal charge
    * partial charge
    * aromatic atom
    * hybridization
    * hydrogen bond donor
    * hydrogen bond acceptor
    * the number of rings the atom belongs to for ring size between 3 and 8

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.
    atom_types : list of str or None
        Atom types to consider for one-hot encoding. If None, we will use a default
        choice of ``'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'``.
    chiral_types : list of Chem.rdchem.ChiralType or None
        Atom chirality to consider for one-hot encoding. If None, we will use a default
        choice of ``Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``.
    hybridization_types : list of Chem.rdchem.HybridizationType or None
        Atom hybridization types to consider for one-hot encoding. If None, we will use a
        default choice of ``Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3``.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import WeaveAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = WeaveAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0418,  0.0000,
                       0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000],
                     [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0402,  0.0000,
                       0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000],
                     [ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.3967,  0.0000,
                       0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000]])}
    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size())
    27

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    PretrainAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h', atom_types=None, chiral_types=None,
                 hybridization_types=None):
        super(WeaveAtomFeaturizer, self).__init__()

        self._atom_data_field = atom_data_field

        if atom_types is None:
            atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        self._atom_types = atom_types

        if chiral_types is None:
            chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
        self._chiral_types = chiral_types

        if hybridization_types is None:
            hybridization_types = [Chem.rdchem.HybridizationType.SP,
                                   Chem.rdchem.HybridizationType.SP2,
                                   Chem.rdchem.HybridizationType.SP3]
        self._hybridization_types = hybridization_types

        self._featurizer = ConcatFeaturizer([
            partial(atom_type_one_hot, allowable_set=atom_types, encode_unknown=True),
            partial(atom_chiral_tag_one_hot, allowable_set=chiral_types),
            atom_formal_charge, atom_partial_charge, atom_is_aromatic,
            partial(atom_hybridization_one_hot, allowable_set=hybridization_types)
        ])

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._atom_data_field]

        return feats.shape[-1]

    def get_donor_acceptor_info(self, mol_feats):
        """Bookkeep whether an atom is donor/acceptor for hydrogen bonds.

        Parameters
        ----------
        mol_feats : tuple of rdkit.Chem.rdMolChemicalFeatures.MolChemicalFeature
            Features for molecules.

        Returns
        -------
        is_donor : dict
            Mapping atom ids to binary values indicating whether atoms
            are donors for hydrogen bonds
        is_acceptor : dict
            Mapping atom ids to binary values indicating whether atoms
            are acceptors for hydrogen bonds
        """
        is_donor = defaultdict(bool)
        is_acceptor = defaultdict(bool)
        # Get hydrogen bond donor/acceptor information
        for feats in mol_feats:
            if feats.GetFamily() == 'Donor':
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_donor[u] = True
            elif feats.GetFamily() == 'Acceptor':
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_acceptor[u] = True

        return is_donor, is_acceptor

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping atom_data_field as specified in the input argument to the atom
            features, which is a float32 tensor of shape (N, M), N is the number of
            atoms and M is the feature size.
        """
        atom_features = []

        AllChem.ComputeGasteigerCharges(mol)
        num_atoms = mol.GetNumAtoms()

        # Get information for donor and acceptor
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        is_donor, is_acceptor = self.get_donor_acceptor_info(mol_feats)

        # Get a symmetrized smallest set of smallest rings
        # Following the practice from Chainer Chemistry (https://github.com/chainer/
        # chainer-chemistry/blob/da2507b38f903a8ee333e487d422ba6dcec49b05/chainer_chemistry/
        # dataset/preprocessors/weavenet_preprocessor.py)
        sssr = Chem.GetSymmSSSR(mol)

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            # Features that can be computed directly from RDKit atom instances, which is a list
            feats = self._featurizer(atom)
            # Donor/acceptor indicator
            feats.append(float(is_donor[i]))
            feats.append(float(is_acceptor[i]))
            # Count the number of rings the atom belongs to for ring size between 3 and 8
            count = [0 for _ in range(3, 9)]
            for ring in sssr:
                ring_size = len(ring)
                if i in ring and 3 <= ring_size <= 8:
                    count[ring_size - 3] += 1
            feats.extend(count)
            atom_features.append(feats)
        atom_features = np.stack(atom_features)

        return {self._atom_data_field: F.zerocopy_from_numpy(atom_features.astype(np.float32))}

class PretrainAtomFeaturizer(object):
    """AtomFeaturizer in Strategies for Pre-training Graph Neural Networks.

    The atom featurization performed in `Strategies for Pre-training Graph Neural Networks
    <https://arxiv.org/abs/1905.12265>`__, which considers:

    * atomic number
    * chirality

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atomic_number_types : list of int or None
        Atomic number types to consider for one-hot encoding. If None, we will use a default
        choice of 1-118.
    chiral_types : list of Chem.rdchem.ChiralType or None
        Atom chirality to consider for one-hot encoding. If None, we will use a default
        choice of ``Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER``.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import PretrainAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = PretrainAtomFeaturizer()
    >>> atom_featurizer(mol)
    {'atomic_number': tensor([5, 5, 7]), 'chirality_type': tensor([0, 0, 0])}

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atomic_number_types=None, chiral_types=None):
        if atomic_number_types is None:
            atomic_number_types = list(range(1, 119))
        self._atomic_number_types = atomic_number_types

        if chiral_types is None:
            chiral_types = [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ]
        self._chiral_types = chiral_types

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping 'atomic_number' and 'chirality_type' to separately an int64 tensor
            of shape (N, 1), N is the number of atoms
        """
        atom_features = []
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_features.append([
                self._atomic_number_types.index(atom.GetAtomicNum()),
                self._chiral_types.index(atom.GetChiralTag())
            ])
        atom_features = np.stack(atom_features)
        atom_features = F.zerocopy_from_numpy(atom_features.astype(np.int64))

        return {
            'atomic_number': atom_features[:, 0],
            'chirality_type': atom_features[:, 1]
        }

class AttentiveFPAtomFeaturizer(BaseAtomFeaturizer):
    """用于在AttentiveFP中对原子进行特征化"""
    """The atom featurizer used in AttentiveFP

    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``B``, ``C``, ``N``, ``O``, ``F``, ``Si``, ``P``, ``S``, ``Cl``, ``As``,
      ``Se``, ``Br``, ``Te``, ``I``, ``At``, and ``other``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 5``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``, and ``other``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.
    * **Whether the atom is chiral center**
    * **One hot encoding of the atom chirality type**. The supported possibilities include
      ``R``, and ``S``.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import AttentiveFPAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                      0., 0., 0.],
                     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                      0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                      0., 0., 0.]])}

    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size('feat'))
    39

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    PretrainAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        ######键是 atom_data_field（默认为 'h'），值是一个 ConcatFeaturizer 类的实例
        ###一个列表，包含了多个特征提取函数
        super(AttentiveFPAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [partial(atom_type_one_hot, allowable_set=[
                    'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S',
                    'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'], encode_unknown=True),
                 partial(atom_degree_one_hot, allowable_set=list(range(6))),
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 partial(atom_hybridization_one_hot, encode_unknown=True),
                 atom_is_aromatic,
                 atom_total_num_H_one_hot,
                 atom_is_chiral_center,
                 atom_chirality_type_one_hot]
            )})

class PAGTNAtomFeaturizer(BaseAtomFeaturizer):
    """The atom featurizer used in PAGTN

    PAGTN is introduced in
    `Path-Augmented Graph Transformer Network. <https://arxiv.org/abs/1905.12712>`__

    The atom features include:

    * **One hot encoding of the atom type**.
    * **One hot encoding of formal charge of the atom**.
    * **One hot encoding of the atom degree**
    * **One hot encoding of explicit valence of an atom**. The supported possibilities
      include ``0 - 6``.
    * **One hot encoding of implicit valence of an atom**. The supported possibilities
      include ``0 - 5``.
    * **Whether the atom is aromatic**.

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import PAGTNAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('C')
    >>> atom_featurizer = PAGTNAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 1., 0., 0.]])}
    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size())
    94

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    PretrainAtomFeaturizer
    WeaveAtomFeaturizer
    AttentiveFPAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                   'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                   'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                   'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                   'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re',
                   'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm',
                   'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', '*', 'UNK']

        super(PAGTNAtomFeaturizer, self).__init__(
            featurizer_funcs={
                atom_data_field: ConcatFeaturizer([partial(atom_type_one_hot,
                                                           allowable_set=SYMBOLS,
                                                           encode_unknown=False),
                                                   atom_formal_charge_one_hot,
                                                   atom_degree_one_hot,
                                                   partial(atom_explicit_valence_one_hot,
                                                           allowable_set=list(range(7)),
                                                           encode_unknown=False),
                                                   partial(atom_implicit_valence_one_hot,
                                                           allowable_set=list(range(6)),
                                                           encode_unknown=False),
                                                   atom_is_aromatic])})

def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """用于对化学键的类型进行独热编码"""
    """One hot encoding for the type of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of Chem.rdchem.BondType
        Bond types to consider. Default: ``Chem.rdchem.BondType.SINGLE``,
        ``Chem.rdchem.BondType.DOUBLE``, ``Chem.rdchem.BondType.TRIPLE``,
        ``Chem.rdchem.BondType.AROMATIC``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
        ###输出键类型进行独热编码的结果
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)

def bond_is_conjugated_one_hot(bond, allowable_set=None, encode_unknown=False):
    """该函数用于对化学键是否是共轭进行独热编码"""
    """One hot encoding for whether the bond is conjugated.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    bond_is_conjugated
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.GetIsConjugated(), allowable_set, encode_unknown)

def bond_is_conjugated(bond):
    """计算键是否是共轭的"""
    """Get whether the bond is conjugated.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    bond_is_conjugated_one_hot
    """
    return [bond.GetIsConjugated()]

def bond_is_in_ring_one_hot(bond, allowable_set=None, encode_unknown=False):
    """用于对化学键是否在任何大小的环中进行独热编码"""
    """One hot encoding for whether the bond is in a ring of any size.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    bond_is_in_ring
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.IsInRing(), allowable_set, encode_unknown)

def bond_is_in_ring(bond):
    """计算键是否在环内"""
    """Get whether the bond is in a ring of any size.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    bond_is_in_ring_one_hot
    """
    return [bond.IsInRing()]

def bond_stereo_one_hot(bond, allowable_set=None, encode_unknown=False):
    """用于对化学键的立体构型进行独热编码"""
    """One hot encoding for the stereo configuration of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of rdkit.Chem.rdchem.BondStereo
        Stereo configurations to consider. Default: ``rdkit.Chem.rdchem.BondStereo.STEREONONE``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOANY``, ``rdkit.Chem.rdchem.BondStereo.STEREOZ``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOE``, ``rdkit.Chem.rdchem.BondStereo.STEREOCIS``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOTRANS``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondStereo.STEREONONE,
                         Chem.rdchem.BondStereo.STEREOANY,
                         Chem.rdchem.BondStereo.STEREOZ,
                         Chem.rdchem.BondStereo.STEREOE,
                         Chem.rdchem.BondStereo.STEREOCIS,
                         Chem.rdchem.BondStereo.STEREOTRANS]
    ###返回化学键的立体构型的独热编码结果
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)

def bond_direction_one_hot(bond, allowable_set=None, encode_unknown=False):
    """用于对化学键的方向进行独热编码"""
    """One hot encoding for the direction of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of Chem.rdchem.BondDir
        Bond directions to consider. Default: ``Chem.rdchem.BondDir.NONE``,
        ``Chem.rdchem.BondDir.ENDUPRIGHT``, ``Chem.rdchem.BondDir.ENDDOWNRIGHT``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondDir.NONE,
                         Chem.rdchem.BondDir.ENDUPRIGHT,
                         Chem.rdchem.BondDir.ENDDOWNRIGHT]
        ###返回化学键的方向的独热编码结果
    return one_hot_encoding(bond.GetBondDir(), allowable_set, encode_unknown)


class BaseBondFeaturizer(object):
    """该类用于分子中的所有化学键进行特征提取"""
    """An abstract class for bond featurizers.
    Loop over all bonds in a molecule and featurize them with the ``featurizer_funcs``.
    We assume the constructed ``DGLGraph`` is a bi-directed graph where the **i** th bond in the
    molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the **(2i)**-th and **(2i+1)**-th edges
    in the DGLGraph.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    featurizer_funcs : dict
        Mapping feature name to the featurization function.
        Each function is of signature ``func(rdkit.Chem.rdchem.Bond) -> list or 1D numpy array``.
    feat_sizes : dict
        Mapping feature name to the size of the corresponding feature. If None, they will be
        computed when needed. Default: None.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops in each bond feature.
        The features of the self loops will be zero except for the additional columns.

    Examples
    --------

    >>> from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = BaseBondFeaturizer({'type': bond_type_one_hot, 'ring': bond_is_in_ring})
    >>> bond_featurizer(mol)
    {'type': tensor([[1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.]]),
     'ring': tensor([[0.], [0.], [0.], [0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('type')
    4
    >>> bond_featurizer.feat_size('ring')
    1

    # Featurization with self loops to add

    >>> bond_featurizer = BaseBondFeaturizer(
    ...                       {'type': bond_type_one_hot, 'ring': bond_is_in_ring},
    ...                       self_loop=True)
    >>> bond_featurizer(mol)
    {'type': tensor([[1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 1.]]),
     'ring': tensor([[0., 0.],
                     [0., 0.],
                     [0., 0.],
                     [0., 0.],
                     [0., 1.],
                     [0., 1.],
                     [0., 1.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('type')
    5
    >>> bond_featurizer.feat_size('ring')
    2

    See Also
    --------
    CanonicalBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, featurizer_funcs, feat_sizes=None, self_loop=False):
        ###featurizer_funcs: 一个字典，将特征名称(键名)映射到用于提取该特征的函数(值)
        ###feat_sizes：一个字典，将特征名称（键名）映射到相应特征的大小（值）
        ###self_loop:一个布尔值，表示是否添加自环，默认是False
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes
        self._self_loop = self_loop

    def feat_size(self, feat_name=None):
        """该函数用于获取键特征（feat_name）的大小"""
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        ###如果未提供键特征的名称，则确保只有一个键特征
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0] ##如果只有一个键特征的话，直接得到名称

        if feat_name not in self.featurizer_funcs:
            #####如果 feat_name 不在 featurizer_funcs 中，会引发 ValueError，指示提供的 feat_name 无效
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        mol = Chem.MolFromSmiles('CCO')
        feats = self(mol)

        return feats[feat_name].shape[1]  ###返回键特征的大小

    def __call__(self, mol):
        """Featurize all bonds in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_bonds = mol.GetNumBonds()  ##获取分子中键的数量
        bond_features = defaultdict(list)

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                bond_features[feat_name].extend([feat, feat.copy()]) ###返回特征对应值列表

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        if self._self_loop and num_bonds > 0: ##如果启用了自环且分子中有键
            num_atoms = mol.GetNumAtoms()
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.cat([feats, torch.zeros(feats.shape[0], 1)], dim=1)
                self_loop_feats = torch.zeros(num_atoms, feats.shape[1])
                self_loop_feats[:, -1] = 1
                feats = torch.cat([feats, self_loop_feats], dim=0)
                processed_features[feat_name] = feats

        if self._self_loop and num_bonds == 0:
            num_atoms = mol.GetNumAtoms()
            toy_mol = Chem.MolFromSmiles('CO')
            processed_features = self(toy_mol)
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.zeros(num_atoms, feats.shape[1])
                feats[:, -1] = 1
                processed_features[feat_name] = feats

        ###返回键特征的字典，对分子中的每个键进行特征化，考虑了自环的情况
        return processed_features

class CanonicalBondFeaturizer(BaseBondFeaturizer):
    """CanonicalBondFeaturizer 是 BaseBondFeaturizer 的子类，用于为分子中的键提取默认的特征。"""
    """A default featurizer for bonds.

    The bond features include:
    * **One hot encoding of the bond type**. The supported bond types include
      ``SINGLE``, ``DOUBLE``, ``TRIPLE``, ``AROMATIC``.
    * **Whether the bond is conjugated.**.
    * **Whether the bond is in a ring of any size.**
    * **One hot encoding of the stereo configuration of a bond**. The supported bond stereo
      configurations include ``STEREONONE``, ``STEREOANY``, ``STEREOZ``, ``STEREOE``,
      ``STEREOCIS``, ``STEREOTRANS``.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops. The feature of the
        self loops will be zero except for the additional column.

    Examples
    --------

    >>> from dgllife.utils import CanonicalBondFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    12

    # Featurization with self loops to add
    >>> bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat', self_loop=True)
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    13

    See Also
    --------
    BaseBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    ###自定义了构造函数，使用super().__init__()函数继承父类的构造函数
    def __init__(self, bond_data_field='e', self_loop=False):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 bond_stereo_one_hot]
            )}, self_loop=self_loop)

# pylint: disable=E1102
class WeaveEdgeFeaturizer(object):
    """Edge featurizer in Weave.

    The edge featurization is introduced in `Molecular Graph Convolutions:
    Moving Beyond Fingerprints <https://arxiv.org/abs/1603.00856>`__.

    This featurization is performed for a complete graph of atoms with self loops added,
    which considers:

    * Number of bonds between each pairs of atoms
    * One-hot encoding of bond type if a bond exists between a pair of atoms
    * Whether a pair of atoms belongs to a same ring

    Parameters
    ----------
    edge_data_field : str
        Name for storing edge features in DGLGraphs, default to ``'e'``.
    max_distance : int
        Maximum number of bonds to consider between each pair of atoms.
        Default to 7.
    bond_types : list of Chem.rdchem.BondType or None
        Bond types to consider for one hot encoding. If None, we consider by
        default single, double, triple and aromatic bonds.

    Examples
    --------

    >>> from dgllife.utils import WeaveEdgeFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CO')
    >>> edge_featurizer = WeaveEdgeFeaturizer(edge_data_field='feat')
    >>> edge_featurizer(mol)
    {'feat': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
    >>> edge_featurizer.feat_size()
    12

    See Also
    --------
    BaseBondFeaturizer
    CanonicalBondFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, edge_data_field='e', max_distance=7, bond_types=None):
        super(WeaveEdgeFeaturizer, self).__init__()

        self._edge_data_field = edge_data_field
        self._max_distance = max_distance
        if bond_types is None:
            bond_types = [Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE,
                          Chem.rdchem.BondType.AROMATIC]
        self._bond_types = bond_types

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._edge_data_field]

        return feats.shape[-1]

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping self._edge_data_field to a float32 tensor of shape (N, M), where
            N is the number of atom pairs and M is the feature size.
        """
        # Part 1 based on number of bonds between each pair of atoms
        distance_matrix = torch.from_numpy(Chem.GetDistanceMatrix(mol))
        # Change shape from (V, V, 1) to (V^2, 1)
        distance_matrix = distance_matrix.float().reshape(-1, 1)
        # Elementwise compare if distance is bigger than 0, 1, ..., max_distance - 1
        distance_indicators = (distance_matrix >
                               torch.arange(0, self._max_distance).float()).float()

        # Part 2 for one hot encoding of bond type.
        num_atoms = mol.GetNumAtoms()
        bond_indicators = torch.zeros(num_atoms, num_atoms, len(self._bond_types))
        for bond in mol.GetBonds():
            bond_type_encoding = torch.tensor(
                bond_type_one_hot(bond, allowable_set=self._bond_types)).float()
            begin_atom_idx, end_atom_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_indicators[begin_atom_idx, end_atom_idx] = bond_type_encoding
            bond_indicators[end_atom_idx, begin_atom_idx] = bond_type_encoding
        # Reshape from (V, V, num_bond_types) to (V^2, num_bond_types)
        bond_indicators = bond_indicators.reshape(-1, len(self._bond_types))

        # Part 3 for whether a pair of atoms belongs to a same ring.
        sssr = Chem.GetSymmSSSR(mol)
        ring_mate_indicators = torch.zeros(num_atoms, num_atoms, 1)
        for ring in sssr:
            ring = list(ring)
            num_atoms_in_ring = len(ring)
            for i in range(num_atoms_in_ring):
                ring_mate_indicators[ring[i], torch.tensor(ring)] = 1
        ring_mate_indicators = ring_mate_indicators.reshape(-1, 1)

        return {self._edge_data_field: torch.cat([distance_indicators,
                                                  bond_indicators,
                                                  ring_mate_indicators], dim=1)}

class PretrainBondFeaturizer(object):
    """BondFeaturizer in Strategies for Pre-training Graph Neural Networks.

    The bond featurization performed in `Strategies for Pre-training Graph Neural Networks
    <https://arxiv.org/abs/1905.12265>`__, which considers:

    * bond type
    * bond direction

    Parameters
    ----------
    bond_types : list of Chem.rdchem.BondType or None
        Bond types to consider. Default to ``Chem.rdchem.BondType.SINGLE``,
        ``Chem.rdchem.BondType.DOUBLE``, ``Chem.rdchem.BondType.TRIPLE``,
        ``Chem.rdchem.BondType.AROMATIC``.
    bond_direction_types : list of Chem.rdchem.BondDir or None
        Bond directions to consider. Default to ``Chem.rdchem.BondDir.NONE``,
        ``Chem.rdchem.BondDir.ENDUPRIGHT``, ``Chem.rdchem.BondDir.ENDDOWNRIGHT``.
    self_loop : bool
        Whether self loops will be added. Default to True.

    Examples
    --------

    >>> from dgllife.utils import PretrainBondFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CO')
    >>> bond_featurizer = PretrainBondFeaturizer()
    >>> bond_featurizer(mol)
    {'bond_type': tensor([0, 0, 4, 4]),
     'bond_direction_type': tensor([0, 0, 0, 0])}
    """
    def __init__(self, bond_types=None, bond_direction_types=None, self_loop=True):
        if bond_types is None:
            bond_types = [
                Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
            ]
        self._bond_types = bond_types

        if bond_direction_types is None:
            bond_direction_types = [
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT
            ]
        self._bond_direction_types = bond_direction_types
        self._self_loop = self_loop

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping 'bond_type' and 'bond_direction_type' separately to an int64
            tensor of shape (N, 1), where N is the number of edges.
        """
        edge_features = []
        num_bonds = mol.GetNumBonds()
        if num_bonds == 0:
            assert self._self_loop, \
                'The molecule has 0 bonds and we should set self._self_loop to True.'

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            bond_feats = [
                self._bond_types.index(bond.GetBondType()),
                self._bond_direction_types.index(bond.GetBondDir())
            ]
            edge_features.extend([bond_feats, bond_feats.copy()])

        if self._self_loop:
            self_loop_features = torch.zeros((mol.GetNumAtoms(), 2), dtype=torch.int64)
            self_loop_features[:, 0] = len(self._bond_types)

        if num_bonds == 0:
            edge_features = self_loop_features
        else:
            edge_features = np.stack(edge_features)
            edge_features = F.zerocopy_from_numpy(edge_features.astype(np.int64))
            if self._self_loop:
                edge_features = torch.cat([edge_features, self_loop_features], dim=0)

        return {'bond_type': edge_features[:, 0], 'bond_direction_type': edge_features[:, 1]}

class AttentiveFPBondFeaturizer(BaseBondFeaturizer):
    """AttentiveFPBondFeaturizer类继承了BaseBondFeaturizer类，用于在 AttentiveFP 模型中对键进行特征提取"""
    """The bond featurizer used in AttentiveFP

    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    The bond features include:
    * **One hot encoding of the bond type**. The supported bond types include
      ``SINGLE``, ``DOUBLE``, ``TRIPLE``, ``AROMATIC``.
    * **Whether the bond is conjugated.**.
    * **Whether the bond is in a ring of any size.**
    * **One hot encoding of the stereo configuration of a bond**. The supported bond stereo
      configurations include ``STEREONONE``, ``STEREOANY``, ``STEREOZ``, ``STEREOE``.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops. The feature of the
        self loops will be zero except for the additional column.

    Examples
    --------

    >>> from dgllife.utils import AttentiveFPBondFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='feat')
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    10

    >>> # Featurization with self loops to add
    >>> bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='feat', self_loop=True)
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    11

    See Also
    --------
    BaseBondFeaturizer
    CanonicalBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, bond_data_field='e', self_loop=False):
        ###调用父类的构造函数
        super(AttentiveFPBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 partial(bond_stereo_one_hot, allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                                             Chem.rdchem.BondStereo.STEREOANY,
                                                             Chem.rdchem.BondStereo.STEREOZ,
                                                             Chem.rdchem.BondStereo.STEREOE])]
            )}, self_loop=self_loop)

class PAGTNEdgeFeaturizer(object):
    """The edge featurizer used in PAGTN

    PAGTN is introduced in
    `Path-Augmented Graph Transformer Network. <https://arxiv.org/abs/1905.12712>`__

    We build a complete graph and the edge features include:
    * **Shortest path between two nodes in terms of bonds**. To encode the path,
        we encode each bond on the path and concatenate their encodings. The encoding
        of a bond contains information about the bond type, whether the bond is
        conjugated and whether the bond is in a ring.
    * **One hot encoding of type of rings based on size and aromaticity**.
    * **One hot encoding of the distance between the nodes**.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_complete_graph` with
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.
    max_length : int
        Maximum distance up to which shortest paths must be considered.
        Paths shorter than max_length will be padded and longer will be
        truncated, default to ``5``.

    Examples
    --------

    >>> from dgllife.utils import PAGTNEdgeFeaturizer
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = PAGTNEdgeFeaturizer(max_length=1)
    >>> bond_featurizer(mol)
    {'e': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size()
    14

    See Also
    --------
    BaseBondFeaturizer
    CanonicalBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    """
    def __init__(self, bond_data_field='e', max_length=5):
        self.bond_data_field = bond_data_field
        # Any two given nodes can belong to the same ring and here only
        # ring sizes of 5 and 6 are used. True & False indicate if it's aromatic or not.
        self.RING_TYPES = [(5, False), (5, True), (6, False), (6, True)]
        self.ordered_pair = lambda a, b: (a, b) if a < b else (b, a)
        self.bond_featurizer = ConcatFeaturizer([bond_type_one_hot,
                                                 bond_is_conjugated,
                                                 bond_is_in_ring])
        self.max_length = max_length

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self.bond_data_field]

        return feats.shape[-1]

    def bond_features(self, mol, path_atoms, ring_info):
        """Computes the edge features for a given pair of nodes.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.
        path_atoms: tuple
            Shortest path between the given pair of nodes.
        ring_info: list
            Different rings that contain the pair of atoms
        """
        features = []
        path_bonds = []
        path_length = len(path_atoms)
        for path_idx in range(path_length - 1):
            bond = mol.GetBondBetweenAtoms(path_atoms[path_idx], path_atoms[path_idx + 1])
            if bond is None:
                import warnings
                warnings.warn('Valid idx of bonds must be passed')
            path_bonds.append(bond)

        for path_idx in range(self.max_length):
            if path_idx < len(path_bonds):
                features.append(self.bond_featurizer(path_bonds[path_idx]))
            else:
                features.append([0, 0, 0, 0, 0, 0])

        if path_length + 1 > self.max_length:
            path_length = self.max_length + 1
        position_feature = np.zeros(self.max_length + 2)
        position_feature[path_length] = 1
        features.append(position_feature)
        if ring_info:
            rfeat = [one_hot_encoding(r, allowable_set=self.RING_TYPES) for r in ring_info]
            rfeat = [True] + np.any(rfeat, axis=0).tolist()
            features.append(rfeat)
        else:
            # This will return a boolean vector with all entries False
            features.append([False] + one_hot_encoding(ring_info, allowable_set=self.RING_TYPES))
        return np.concatenate(features, axis=0)

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping self._edge_data_field to a float32 tensor of shape (N, M), where
            N is the number of atom pairs and M is the feature size depending on max_length.
        """

        n_atoms = mol.GetNumAtoms()
        # To get the shortest paths between two nodes.
        paths_dict = {
            (i, j): Chem.rdmolops.GetShortestPath(mol, i, j)
            for i in range(n_atoms)
            for j in range(n_atoms)
            if i != j
            }
        # To get info if two nodes belong to the same ring.
        rings_dict = {}
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        for ring in ssr:
            ring_sz = len(ring)
            is_aromatic = True
            for atom_idx in ring:
                if not mol.GetAtoms()[atom_idx].GetIsAromatic():
                    is_aromatic = False
                    break
            for ring_idx, atom_idx in enumerate(ring):
                for other_idx in ring[ring_idx:]:
                    atom_pair = self.ordered_pair(atom_idx, other_idx)
                    if atom_pair not in rings_dict:
                        rings_dict[atom_pair] = [(ring_sz, is_aromatic)]
                    else:
                        if (ring_sz, is_aromatic) not in rings_dict[atom_pair]:
                            rings_dict[atom_pair].append((ring_sz, is_aromatic))
        # Featurizer
        feats = []
        for i in range(n_atoms):
            for j in range(n_atoms):

                if (i, j) not in paths_dict:
                    feats.append(np.zeros(7*self.max_length + 7))
                    continue
                ring_info = rings_dict.get(self.ordered_pair(i, j), [])
                feats.append(self.bond_features(mol, paths_dict[(i, j)], ring_info))

        return {self.bond_data_field: torch.tensor(feats).float()}






