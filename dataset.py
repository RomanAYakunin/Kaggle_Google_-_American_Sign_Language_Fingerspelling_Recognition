import os
import shutil
import sys
import math
import torch
import json
import numpy as np
import torch.nn as nn
import polars as pl
from torch.utils.data import Dataset, DataLoader
from utils import get_paths, get_phrases
from utils import save_arrs, load_arrs
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
from utils import get_meta, phrases_to_labels
from copy import deepcopy
import functools

POINTS_PER_FRAME = 543


class FeatureGenerator(nn.Module):
    def __init__(self):
        super(FeatureGenerator, self).__init__()
        feature_dict = dict(
            upper_lip_outline=np.array([0, 13, 37, 39, 40, 61, 78, 80, 81, 82, 185, 191,
                                        267, 269, 270, 291, 308, 310, 311, 312, 409, 415]),
            lower_lip_outline=np.array([14, 17, 84, 87, 88, 91, 95, 146, 178, 181,
                                        314, 317, 318, 321, 324, 375, 402, 405]),
            pose=np.array([489, 493, 494, 495, 497, 499, 501, 503, 505, 507, 509, 511, 513, 515,
                                490, 491, 492, 496, 498, 500, 502, 504, 506, 508, 510, 512, 514]),
            left_hand=np.arange(468, 489),
            right_hand=np.arange(522, 543)
        )
        reflect_dict = dict(
            upper_lip_outline=np.array([0, 13, 267, 269, 270, 291, 308, 310, 311, 312, 409, 415,
                                        37, 39, 40, 61, 78, 80, 81, 82, 185, 191]),
            lower_lip_outline=np.array([14, 17, 314, 317, 318, 321, 324, 375, 402, 405,
                                        84, 87, 88, 91, 95, 146, 178, 181]),
            pose=[489, 490, 491, 492, 496, 498, 500, 502, 504, 506, 508, 510, 512, 514,
                       493, 494, 495, 497, 499, 501, 503, 505, 507, 509, 511, 513, 515],
            left_hand=feature_dict['right_hand'],
            right_hand=feature_dict['left_hand']
        )
        feature_type_list = ['upper_lip_outline', 'lower_lip_outline', 'pose', 'left_hand', 'right_hand']
        point_arr = []
        reflect_arr = []
        feature_range_dict = dict()
        curr_idx = 0
        for feature_type in feature_type_list:
            point_arr.append(feature_dict[feature_type])
            reflect_arr.append(reflect_dict[feature_type])
            feature_range_dict[feature_type] = (curr_idx, curr_idx + len(feature_dict[feature_type]))
            curr_idx += len(feature_dict[feature_type])
        self.norm_ranges = [
            (feature_range_dict['upper_lip_outline'][0], feature_range_dict['lower_lip_outline'][1]),
            feature_range_dict['pose'],
            feature_range_dict['left_hand'],
            feature_range_dict['right_hand']
        ]
        point_arr = np.concatenate(point_arr)
        reflect_arr = np.concatenate(reflect_arr)
        idx_map = np.zeros(POINTS_PER_FRAME)
        idx_map[point_arr] = np.arange(len(point_arr))
        reflect_arr = idx_map[reflect_arr]
        point_arr = torch.Tensor(point_arr).type(torch.int32)
        self.register_buffer('point_arr', point_arr)
        reflect_arr = torch.Tensor(reflect_arr).type(torch.long)
        self.register_buffer('reflect_arr', reflect_arr)
        self.num_points = len(self.point_arr)
        self.num_axes = 2
        self.max_len = 800
        self.nan_value = 0


def get_column_names(filter_columns=True):
    columns = np.array(['x_face_0', 'x_face_1', 'x_face_2', 'x_face_3', 'x_face_4', 'x_face_5', 'x_face_6', 'x_face_7', 'x_face_8', 'x_face_9', 'x_face_10', 'x_face_11', 'x_face_12', 'x_face_13', 'x_face_14', 'x_face_15', 'x_face_16', 'x_face_17', 'x_face_18', 'x_face_19', 'x_face_20', 'x_face_21', 'x_face_22', 'x_face_23', 'x_face_24', 'x_face_25', 'x_face_26', 'x_face_27', 'x_face_28', 'x_face_29', 'x_face_30', 'x_face_31', 'x_face_32', 'x_face_33', 'x_face_34', 'x_face_35', 'x_face_36', 'x_face_37', 'x_face_38', 'x_face_39', 'x_face_40', 'x_face_41', 'x_face_42', 'x_face_43', 'x_face_44', 'x_face_45', 'x_face_46', 'x_face_47', 'x_face_48', 'x_face_49', 'x_face_50', 'x_face_51', 'x_face_52', 'x_face_53', 'x_face_54', 'x_face_55', 'x_face_56', 'x_face_57', 'x_face_58', 'x_face_59', 'x_face_60', 'x_face_61', 'x_face_62', 'x_face_63', 'x_face_64', 'x_face_65', 'x_face_66', 'x_face_67', 'x_face_68', 'x_face_69', 'x_face_70', 'x_face_71', 'x_face_72', 'x_face_73', 'x_face_74', 'x_face_75', 'x_face_76', 'x_face_77', 'x_face_78', 'x_face_79', 'x_face_80', 'x_face_81', 'x_face_82', 'x_face_83', 'x_face_84', 'x_face_85', 'x_face_86', 'x_face_87', 'x_face_88', 'x_face_89', 'x_face_90', 'x_face_91', 'x_face_92', 'x_face_93', 'x_face_94', 'x_face_95', 'x_face_96', 'x_face_97', 'x_face_98', 'x_face_99', 'x_face_100', 'x_face_101', 'x_face_102', 'x_face_103', 'x_face_104', 'x_face_105', 'x_face_106', 'x_face_107', 'x_face_108', 'x_face_109', 'x_face_110', 'x_face_111', 'x_face_112', 'x_face_113', 'x_face_114', 'x_face_115', 'x_face_116', 'x_face_117', 'x_face_118', 'x_face_119', 'x_face_120', 'x_face_121', 'x_face_122', 'x_face_123', 'x_face_124', 'x_face_125', 'x_face_126', 'x_face_127', 'x_face_128', 'x_face_129', 'x_face_130', 'x_face_131', 'x_face_132', 'x_face_133', 'x_face_134', 'x_face_135', 'x_face_136', 'x_face_137', 'x_face_138', 'x_face_139', 'x_face_140', 'x_face_141', 'x_face_142', 'x_face_143', 'x_face_144', 'x_face_145', 'x_face_146', 'x_face_147', 'x_face_148', 'x_face_149', 'x_face_150', 'x_face_151', 'x_face_152', 'x_face_153', 'x_face_154', 'x_face_155', 'x_face_156', 'x_face_157', 'x_face_158', 'x_face_159', 'x_face_160', 'x_face_161', 'x_face_162', 'x_face_163', 'x_face_164', 'x_face_165', 'x_face_166', 'x_face_167', 'x_face_168', 'x_face_169', 'x_face_170', 'x_face_171', 'x_face_172', 'x_face_173', 'x_face_174', 'x_face_175', 'x_face_176', 'x_face_177', 'x_face_178', 'x_face_179', 'x_face_180', 'x_face_181', 'x_face_182', 'x_face_183', 'x_face_184', 'x_face_185', 'x_face_186', 'x_face_187', 'x_face_188', 'x_face_189', 'x_face_190', 'x_face_191', 'x_face_192', 'x_face_193', 'x_face_194', 'x_face_195', 'x_face_196', 'x_face_197', 'x_face_198', 'x_face_199', 'x_face_200', 'x_face_201', 'x_face_202', 'x_face_203', 'x_face_204', 'x_face_205', 'x_face_206', 'x_face_207', 'x_face_208', 'x_face_209', 'x_face_210', 'x_face_211', 'x_face_212', 'x_face_213', 'x_face_214', 'x_face_215', 'x_face_216', 'x_face_217', 'x_face_218', 'x_face_219', 'x_face_220', 'x_face_221', 'x_face_222', 'x_face_223', 'x_face_224', 'x_face_225', 'x_face_226', 'x_face_227', 'x_face_228', 'x_face_229', 'x_face_230', 'x_face_231', 'x_face_232', 'x_face_233', 'x_face_234', 'x_face_235', 'x_face_236', 'x_face_237', 'x_face_238', 'x_face_239', 'x_face_240', 'x_face_241', 'x_face_242', 'x_face_243', 'x_face_244', 'x_face_245', 'x_face_246', 'x_face_247', 'x_face_248', 'x_face_249', 'x_face_250', 'x_face_251', 'x_face_252', 'x_face_253', 'x_face_254', 'x_face_255', 'x_face_256', 'x_face_257', 'x_face_258', 'x_face_259', 'x_face_260', 'x_face_261', 'x_face_262', 'x_face_263', 'x_face_264', 'x_face_265', 'x_face_266', 'x_face_267', 'x_face_268', 'x_face_269', 'x_face_270', 'x_face_271', 'x_face_272', 'x_face_273', 'x_face_274', 'x_face_275', 'x_face_276', 'x_face_277', 'x_face_278', 'x_face_279', 'x_face_280', 'x_face_281', 'x_face_282', 'x_face_283', 'x_face_284', 'x_face_285', 'x_face_286', 'x_face_287', 'x_face_288', 'x_face_289', 'x_face_290', 'x_face_291', 'x_face_292', 'x_face_293', 'x_face_294', 'x_face_295', 'x_face_296', 'x_face_297', 'x_face_298', 'x_face_299', 'x_face_300', 'x_face_301', 'x_face_302', 'x_face_303', 'x_face_304', 'x_face_305', 'x_face_306', 'x_face_307', 'x_face_308', 'x_face_309', 'x_face_310', 'x_face_311', 'x_face_312', 'x_face_313', 'x_face_314', 'x_face_315', 'x_face_316', 'x_face_317', 'x_face_318', 'x_face_319', 'x_face_320', 'x_face_321', 'x_face_322', 'x_face_323', 'x_face_324', 'x_face_325', 'x_face_326', 'x_face_327', 'x_face_328', 'x_face_329', 'x_face_330', 'x_face_331', 'x_face_332', 'x_face_333', 'x_face_334', 'x_face_335', 'x_face_336', 'x_face_337', 'x_face_338', 'x_face_339', 'x_face_340', 'x_face_341', 'x_face_342', 'x_face_343', 'x_face_344', 'x_face_345', 'x_face_346', 'x_face_347', 'x_face_348', 'x_face_349', 'x_face_350', 'x_face_351', 'x_face_352', 'x_face_353', 'x_face_354', 'x_face_355', 'x_face_356', 'x_face_357', 'x_face_358', 'x_face_359', 'x_face_360', 'x_face_361', 'x_face_362', 'x_face_363', 'x_face_364', 'x_face_365', 'x_face_366', 'x_face_367', 'x_face_368', 'x_face_369', 'x_face_370', 'x_face_371', 'x_face_372', 'x_face_373', 'x_face_374', 'x_face_375', 'x_face_376', 'x_face_377', 'x_face_378', 'x_face_379', 'x_face_380', 'x_face_381', 'x_face_382', 'x_face_383', 'x_face_384', 'x_face_385', 'x_face_386', 'x_face_387', 'x_face_388', 'x_face_389', 'x_face_390', 'x_face_391', 'x_face_392', 'x_face_393', 'x_face_394', 'x_face_395', 'x_face_396', 'x_face_397', 'x_face_398', 'x_face_399', 'x_face_400', 'x_face_401', 'x_face_402', 'x_face_403', 'x_face_404', 'x_face_405', 'x_face_406', 'x_face_407', 'x_face_408', 'x_face_409', 'x_face_410', 'x_face_411', 'x_face_412', 'x_face_413', 'x_face_414', 'x_face_415', 'x_face_416', 'x_face_417', 'x_face_418', 'x_face_419', 'x_face_420', 'x_face_421', 'x_face_422', 'x_face_423', 'x_face_424', 'x_face_425', 'x_face_426', 'x_face_427', 'x_face_428', 'x_face_429', 'x_face_430', 'x_face_431', 'x_face_432', 'x_face_433', 'x_face_434', 'x_face_435', 'x_face_436', 'x_face_437', 'x_face_438', 'x_face_439', 'x_face_440', 'x_face_441', 'x_face_442', 'x_face_443', 'x_face_444', 'x_face_445', 'x_face_446', 'x_face_447', 'x_face_448', 'x_face_449', 'x_face_450', 'x_face_451', 'x_face_452', 'x_face_453', 'x_face_454', 'x_face_455', 'x_face_456', 'x_face_457', 'x_face_458', 'x_face_459', 'x_face_460', 'x_face_461', 'x_face_462', 'x_face_463', 'x_face_464', 'x_face_465', 'x_face_466', 'x_face_467', 'x_left_hand_0', 'x_left_hand_1', 'x_left_hand_2', 'x_left_hand_3', 'x_left_hand_4', 'x_left_hand_5', 'x_left_hand_6', 'x_left_hand_7', 'x_left_hand_8', 'x_left_hand_9', 'x_left_hand_10', 'x_left_hand_11', 'x_left_hand_12', 'x_left_hand_13', 'x_left_hand_14', 'x_left_hand_15', 'x_left_hand_16', 'x_left_hand_17', 'x_left_hand_18', 'x_left_hand_19', 'x_left_hand_20', 'x_pose_0', 'x_pose_1', 'x_pose_2', 'x_pose_3', 'x_pose_4', 'x_pose_5', 'x_pose_6', 'x_pose_7', 'x_pose_8', 'x_pose_9', 'x_pose_10', 'x_pose_11', 'x_pose_12', 'x_pose_13', 'x_pose_14', 'x_pose_15', 'x_pose_16', 'x_pose_17', 'x_pose_18', 'x_pose_19', 'x_pose_20', 'x_pose_21', 'x_pose_22', 'x_pose_23', 'x_pose_24', 'x_pose_25', 'x_pose_26', 'x_pose_27', 'x_pose_28', 'x_pose_29', 'x_pose_30', 'x_pose_31', 'x_pose_32', 'x_right_hand_0', 'x_right_hand_1', 'x_right_hand_2', 'x_right_hand_3', 'x_right_hand_4', 'x_right_hand_5', 'x_right_hand_6', 'x_right_hand_7', 'x_right_hand_8', 'x_right_hand_9', 'x_right_hand_10', 'x_right_hand_11', 'x_right_hand_12', 'x_right_hand_13', 'x_right_hand_14', 'x_right_hand_15', 'x_right_hand_16', 'x_right_hand_17', 'x_right_hand_18', 'x_right_hand_19', 'x_right_hand_20', 'y_face_0', 'y_face_1', 'y_face_2', 'y_face_3', 'y_face_4', 'y_face_5', 'y_face_6', 'y_face_7', 'y_face_8', 'y_face_9', 'y_face_10', 'y_face_11', 'y_face_12', 'y_face_13', 'y_face_14', 'y_face_15', 'y_face_16', 'y_face_17', 'y_face_18', 'y_face_19', 'y_face_20', 'y_face_21', 'y_face_22', 'y_face_23', 'y_face_24', 'y_face_25', 'y_face_26', 'y_face_27', 'y_face_28', 'y_face_29', 'y_face_30', 'y_face_31', 'y_face_32', 'y_face_33', 'y_face_34', 'y_face_35', 'y_face_36', 'y_face_37', 'y_face_38', 'y_face_39', 'y_face_40', 'y_face_41', 'y_face_42', 'y_face_43', 'y_face_44', 'y_face_45', 'y_face_46', 'y_face_47', 'y_face_48', 'y_face_49', 'y_face_50', 'y_face_51', 'y_face_52', 'y_face_53', 'y_face_54', 'y_face_55', 'y_face_56', 'y_face_57', 'y_face_58', 'y_face_59', 'y_face_60', 'y_face_61', 'y_face_62', 'y_face_63', 'y_face_64', 'y_face_65', 'y_face_66', 'y_face_67', 'y_face_68', 'y_face_69', 'y_face_70', 'y_face_71', 'y_face_72', 'y_face_73', 'y_face_74', 'y_face_75', 'y_face_76', 'y_face_77', 'y_face_78', 'y_face_79', 'y_face_80', 'y_face_81', 'y_face_82', 'y_face_83', 'y_face_84', 'y_face_85', 'y_face_86', 'y_face_87', 'y_face_88', 'y_face_89', 'y_face_90', 'y_face_91', 'y_face_92', 'y_face_93', 'y_face_94', 'y_face_95', 'y_face_96', 'y_face_97', 'y_face_98', 'y_face_99', 'y_face_100', 'y_face_101', 'y_face_102', 'y_face_103', 'y_face_104', 'y_face_105', 'y_face_106', 'y_face_107', 'y_face_108', 'y_face_109', 'y_face_110', 'y_face_111', 'y_face_112', 'y_face_113', 'y_face_114', 'y_face_115', 'y_face_116', 'y_face_117', 'y_face_118', 'y_face_119', 'y_face_120', 'y_face_121', 'y_face_122', 'y_face_123', 'y_face_124', 'y_face_125', 'y_face_126', 'y_face_127', 'y_face_128', 'y_face_129', 'y_face_130', 'y_face_131', 'y_face_132', 'y_face_133', 'y_face_134', 'y_face_135', 'y_face_136', 'y_face_137', 'y_face_138', 'y_face_139', 'y_face_140', 'y_face_141', 'y_face_142', 'y_face_143', 'y_face_144', 'y_face_145', 'y_face_146', 'y_face_147', 'y_face_148', 'y_face_149', 'y_face_150', 'y_face_151', 'y_face_152', 'y_face_153', 'y_face_154', 'y_face_155', 'y_face_156', 'y_face_157', 'y_face_158', 'y_face_159', 'y_face_160', 'y_face_161', 'y_face_162', 'y_face_163', 'y_face_164', 'y_face_165', 'y_face_166', 'y_face_167', 'y_face_168', 'y_face_169', 'y_face_170', 'y_face_171', 'y_face_172', 'y_face_173', 'y_face_174', 'y_face_175', 'y_face_176', 'y_face_177', 'y_face_178', 'y_face_179', 'y_face_180', 'y_face_181', 'y_face_182', 'y_face_183', 'y_face_184', 'y_face_185', 'y_face_186', 'y_face_187', 'y_face_188', 'y_face_189', 'y_face_190', 'y_face_191', 'y_face_192', 'y_face_193', 'y_face_194', 'y_face_195', 'y_face_196', 'y_face_197', 'y_face_198', 'y_face_199', 'y_face_200', 'y_face_201', 'y_face_202', 'y_face_203', 'y_face_204', 'y_face_205', 'y_face_206', 'y_face_207', 'y_face_208', 'y_face_209', 'y_face_210', 'y_face_211', 'y_face_212', 'y_face_213', 'y_face_214', 'y_face_215', 'y_face_216', 'y_face_217', 'y_face_218', 'y_face_219', 'y_face_220', 'y_face_221', 'y_face_222', 'y_face_223', 'y_face_224', 'y_face_225', 'y_face_226', 'y_face_227', 'y_face_228', 'y_face_229', 'y_face_230', 'y_face_231', 'y_face_232', 'y_face_233', 'y_face_234', 'y_face_235', 'y_face_236', 'y_face_237', 'y_face_238', 'y_face_239', 'y_face_240', 'y_face_241', 'y_face_242', 'y_face_243', 'y_face_244', 'y_face_245', 'y_face_246', 'y_face_247', 'y_face_248', 'y_face_249', 'y_face_250', 'y_face_251', 'y_face_252', 'y_face_253', 'y_face_254', 'y_face_255', 'y_face_256', 'y_face_257', 'y_face_258', 'y_face_259', 'y_face_260', 'y_face_261', 'y_face_262', 'y_face_263', 'y_face_264', 'y_face_265', 'y_face_266', 'y_face_267', 'y_face_268', 'y_face_269', 'y_face_270', 'y_face_271', 'y_face_272', 'y_face_273', 'y_face_274', 'y_face_275', 'y_face_276', 'y_face_277', 'y_face_278', 'y_face_279', 'y_face_280', 'y_face_281', 'y_face_282', 'y_face_283', 'y_face_284', 'y_face_285', 'y_face_286', 'y_face_287', 'y_face_288', 'y_face_289', 'y_face_290', 'y_face_291', 'y_face_292', 'y_face_293', 'y_face_294', 'y_face_295', 'y_face_296', 'y_face_297', 'y_face_298', 'y_face_299', 'y_face_300', 'y_face_301', 'y_face_302', 'y_face_303', 'y_face_304', 'y_face_305', 'y_face_306', 'y_face_307', 'y_face_308', 'y_face_309', 'y_face_310', 'y_face_311', 'y_face_312', 'y_face_313', 'y_face_314', 'y_face_315', 'y_face_316', 'y_face_317', 'y_face_318', 'y_face_319', 'y_face_320', 'y_face_321', 'y_face_322', 'y_face_323', 'y_face_324', 'y_face_325', 'y_face_326', 'y_face_327', 'y_face_328', 'y_face_329', 'y_face_330', 'y_face_331', 'y_face_332', 'y_face_333', 'y_face_334', 'y_face_335', 'y_face_336', 'y_face_337', 'y_face_338', 'y_face_339', 'y_face_340', 'y_face_341', 'y_face_342', 'y_face_343', 'y_face_344', 'y_face_345', 'y_face_346', 'y_face_347', 'y_face_348', 'y_face_349', 'y_face_350', 'y_face_351', 'y_face_352', 'y_face_353', 'y_face_354', 'y_face_355', 'y_face_356', 'y_face_357', 'y_face_358', 'y_face_359', 'y_face_360', 'y_face_361', 'y_face_362', 'y_face_363', 'y_face_364', 'y_face_365', 'y_face_366', 'y_face_367', 'y_face_368', 'y_face_369', 'y_face_370', 'y_face_371', 'y_face_372', 'y_face_373', 'y_face_374', 'y_face_375', 'y_face_376', 'y_face_377', 'y_face_378', 'y_face_379', 'y_face_380', 'y_face_381', 'y_face_382', 'y_face_383', 'y_face_384', 'y_face_385', 'y_face_386', 'y_face_387', 'y_face_388', 'y_face_389', 'y_face_390', 'y_face_391', 'y_face_392', 'y_face_393', 'y_face_394', 'y_face_395', 'y_face_396', 'y_face_397', 'y_face_398', 'y_face_399', 'y_face_400', 'y_face_401', 'y_face_402', 'y_face_403', 'y_face_404', 'y_face_405', 'y_face_406', 'y_face_407', 'y_face_408', 'y_face_409', 'y_face_410', 'y_face_411', 'y_face_412', 'y_face_413', 'y_face_414', 'y_face_415', 'y_face_416', 'y_face_417', 'y_face_418', 'y_face_419', 'y_face_420', 'y_face_421', 'y_face_422', 'y_face_423', 'y_face_424', 'y_face_425', 'y_face_426', 'y_face_427', 'y_face_428', 'y_face_429', 'y_face_430', 'y_face_431', 'y_face_432', 'y_face_433', 'y_face_434', 'y_face_435', 'y_face_436', 'y_face_437', 'y_face_438', 'y_face_439', 'y_face_440', 'y_face_441', 'y_face_442', 'y_face_443', 'y_face_444', 'y_face_445', 'y_face_446', 'y_face_447', 'y_face_448', 'y_face_449', 'y_face_450', 'y_face_451', 'y_face_452', 'y_face_453', 'y_face_454', 'y_face_455', 'y_face_456', 'y_face_457', 'y_face_458', 'y_face_459', 'y_face_460', 'y_face_461', 'y_face_462', 'y_face_463', 'y_face_464', 'y_face_465', 'y_face_466', 'y_face_467', 'y_left_hand_0', 'y_left_hand_1', 'y_left_hand_2', 'y_left_hand_3', 'y_left_hand_4', 'y_left_hand_5', 'y_left_hand_6', 'y_left_hand_7', 'y_left_hand_8', 'y_left_hand_9', 'y_left_hand_10', 'y_left_hand_11', 'y_left_hand_12', 'y_left_hand_13', 'y_left_hand_14', 'y_left_hand_15', 'y_left_hand_16', 'y_left_hand_17', 'y_left_hand_18', 'y_left_hand_19', 'y_left_hand_20', 'y_pose_0', 'y_pose_1', 'y_pose_2', 'y_pose_3', 'y_pose_4', 'y_pose_5', 'y_pose_6', 'y_pose_7', 'y_pose_8', 'y_pose_9', 'y_pose_10', 'y_pose_11', 'y_pose_12', 'y_pose_13', 'y_pose_14', 'y_pose_15', 'y_pose_16', 'y_pose_17', 'y_pose_18', 'y_pose_19', 'y_pose_20', 'y_pose_21', 'y_pose_22', 'y_pose_23', 'y_pose_24', 'y_pose_25', 'y_pose_26', 'y_pose_27', 'y_pose_28', 'y_pose_29', 'y_pose_30', 'y_pose_31', 'y_pose_32', 'y_right_hand_0', 'y_right_hand_1', 'y_right_hand_2', 'y_right_hand_3', 'y_right_hand_4', 'y_right_hand_5', 'y_right_hand_6', 'y_right_hand_7', 'y_right_hand_8', 'y_right_hand_9', 'y_right_hand_10', 'y_right_hand_11', 'y_right_hand_12', 'y_right_hand_13', 'y_right_hand_14', 'y_right_hand_15', 'y_right_hand_16', 'y_right_hand_17', 'y_right_hand_18', 'y_right_hand_19', 'y_right_hand_20', 'z_face_0', 'z_face_1', 'z_face_2', 'z_face_3', 'z_face_4', 'z_face_5', 'z_face_6', 'z_face_7', 'z_face_8', 'z_face_9', 'z_face_10', 'z_face_11', 'z_face_12', 'z_face_13', 'z_face_14', 'z_face_15', 'z_face_16', 'z_face_17', 'z_face_18', 'z_face_19', 'z_face_20', 'z_face_21', 'z_face_22', 'z_face_23', 'z_face_24', 'z_face_25', 'z_face_26', 'z_face_27', 'z_face_28', 'z_face_29', 'z_face_30', 'z_face_31', 'z_face_32', 'z_face_33', 'z_face_34', 'z_face_35', 'z_face_36', 'z_face_37', 'z_face_38', 'z_face_39', 'z_face_40', 'z_face_41', 'z_face_42', 'z_face_43', 'z_face_44', 'z_face_45', 'z_face_46', 'z_face_47', 'z_face_48', 'z_face_49', 'z_face_50', 'z_face_51', 'z_face_52', 'z_face_53', 'z_face_54', 'z_face_55', 'z_face_56', 'z_face_57', 'z_face_58', 'z_face_59', 'z_face_60', 'z_face_61', 'z_face_62', 'z_face_63', 'z_face_64', 'z_face_65', 'z_face_66', 'z_face_67', 'z_face_68', 'z_face_69', 'z_face_70', 'z_face_71', 'z_face_72', 'z_face_73', 'z_face_74', 'z_face_75', 'z_face_76', 'z_face_77', 'z_face_78', 'z_face_79', 'z_face_80', 'z_face_81', 'z_face_82', 'z_face_83', 'z_face_84', 'z_face_85', 'z_face_86', 'z_face_87', 'z_face_88', 'z_face_89', 'z_face_90', 'z_face_91', 'z_face_92', 'z_face_93', 'z_face_94', 'z_face_95', 'z_face_96', 'z_face_97', 'z_face_98', 'z_face_99', 'z_face_100', 'z_face_101', 'z_face_102', 'z_face_103', 'z_face_104', 'z_face_105', 'z_face_106', 'z_face_107', 'z_face_108', 'z_face_109', 'z_face_110', 'z_face_111', 'z_face_112', 'z_face_113', 'z_face_114', 'z_face_115', 'z_face_116', 'z_face_117', 'z_face_118', 'z_face_119', 'z_face_120', 'z_face_121', 'z_face_122', 'z_face_123', 'z_face_124', 'z_face_125', 'z_face_126', 'z_face_127', 'z_face_128', 'z_face_129', 'z_face_130', 'z_face_131', 'z_face_132', 'z_face_133', 'z_face_134', 'z_face_135', 'z_face_136', 'z_face_137', 'z_face_138', 'z_face_139', 'z_face_140', 'z_face_141', 'z_face_142', 'z_face_143', 'z_face_144', 'z_face_145', 'z_face_146', 'z_face_147', 'z_face_148', 'z_face_149', 'z_face_150', 'z_face_151', 'z_face_152', 'z_face_153', 'z_face_154', 'z_face_155', 'z_face_156', 'z_face_157', 'z_face_158', 'z_face_159', 'z_face_160', 'z_face_161', 'z_face_162', 'z_face_163', 'z_face_164', 'z_face_165', 'z_face_166', 'z_face_167', 'z_face_168', 'z_face_169', 'z_face_170', 'z_face_171', 'z_face_172', 'z_face_173', 'z_face_174', 'z_face_175', 'z_face_176', 'z_face_177', 'z_face_178', 'z_face_179', 'z_face_180', 'z_face_181', 'z_face_182', 'z_face_183', 'z_face_184', 'z_face_185', 'z_face_186', 'z_face_187', 'z_face_188', 'z_face_189', 'z_face_190', 'z_face_191', 'z_face_192', 'z_face_193', 'z_face_194', 'z_face_195', 'z_face_196', 'z_face_197', 'z_face_198', 'z_face_199', 'z_face_200', 'z_face_201', 'z_face_202', 'z_face_203', 'z_face_204', 'z_face_205', 'z_face_206', 'z_face_207', 'z_face_208', 'z_face_209', 'z_face_210', 'z_face_211', 'z_face_212', 'z_face_213', 'z_face_214', 'z_face_215', 'z_face_216', 'z_face_217', 'z_face_218', 'z_face_219', 'z_face_220', 'z_face_221', 'z_face_222', 'z_face_223', 'z_face_224', 'z_face_225', 'z_face_226', 'z_face_227', 'z_face_228', 'z_face_229', 'z_face_230', 'z_face_231', 'z_face_232', 'z_face_233', 'z_face_234', 'z_face_235', 'z_face_236', 'z_face_237', 'z_face_238', 'z_face_239', 'z_face_240', 'z_face_241', 'z_face_242', 'z_face_243', 'z_face_244', 'z_face_245', 'z_face_246', 'z_face_247', 'z_face_248', 'z_face_249', 'z_face_250', 'z_face_251', 'z_face_252', 'z_face_253', 'z_face_254', 'z_face_255', 'z_face_256', 'z_face_257', 'z_face_258', 'z_face_259', 'z_face_260', 'z_face_261', 'z_face_262', 'z_face_263', 'z_face_264', 'z_face_265', 'z_face_266', 'z_face_267', 'z_face_268', 'z_face_269', 'z_face_270', 'z_face_271', 'z_face_272', 'z_face_273', 'z_face_274', 'z_face_275', 'z_face_276', 'z_face_277', 'z_face_278', 'z_face_279', 'z_face_280', 'z_face_281', 'z_face_282', 'z_face_283', 'z_face_284', 'z_face_285', 'z_face_286', 'z_face_287', 'z_face_288', 'z_face_289', 'z_face_290', 'z_face_291', 'z_face_292', 'z_face_293', 'z_face_294', 'z_face_295', 'z_face_296', 'z_face_297', 'z_face_298', 'z_face_299', 'z_face_300', 'z_face_301', 'z_face_302', 'z_face_303', 'z_face_304', 'z_face_305', 'z_face_306', 'z_face_307', 'z_face_308', 'z_face_309', 'z_face_310', 'z_face_311', 'z_face_312', 'z_face_313', 'z_face_314', 'z_face_315', 'z_face_316', 'z_face_317', 'z_face_318', 'z_face_319', 'z_face_320', 'z_face_321', 'z_face_322', 'z_face_323', 'z_face_324', 'z_face_325', 'z_face_326', 'z_face_327', 'z_face_328', 'z_face_329', 'z_face_330', 'z_face_331', 'z_face_332', 'z_face_333', 'z_face_334', 'z_face_335', 'z_face_336', 'z_face_337', 'z_face_338', 'z_face_339', 'z_face_340', 'z_face_341', 'z_face_342', 'z_face_343', 'z_face_344', 'z_face_345', 'z_face_346', 'z_face_347', 'z_face_348', 'z_face_349', 'z_face_350', 'z_face_351', 'z_face_352', 'z_face_353', 'z_face_354', 'z_face_355', 'z_face_356', 'z_face_357', 'z_face_358', 'z_face_359', 'z_face_360', 'z_face_361', 'z_face_362', 'z_face_363', 'z_face_364', 'z_face_365', 'z_face_366', 'z_face_367', 'z_face_368', 'z_face_369', 'z_face_370', 'z_face_371', 'z_face_372', 'z_face_373', 'z_face_374', 'z_face_375', 'z_face_376', 'z_face_377', 'z_face_378', 'z_face_379', 'z_face_380', 'z_face_381', 'z_face_382', 'z_face_383', 'z_face_384', 'z_face_385', 'z_face_386', 'z_face_387', 'z_face_388', 'z_face_389', 'z_face_390', 'z_face_391', 'z_face_392', 'z_face_393', 'z_face_394', 'z_face_395', 'z_face_396', 'z_face_397', 'z_face_398', 'z_face_399', 'z_face_400', 'z_face_401', 'z_face_402', 'z_face_403', 'z_face_404', 'z_face_405', 'z_face_406', 'z_face_407', 'z_face_408', 'z_face_409', 'z_face_410', 'z_face_411', 'z_face_412', 'z_face_413', 'z_face_414', 'z_face_415', 'z_face_416', 'z_face_417', 'z_face_418', 'z_face_419', 'z_face_420', 'z_face_421', 'z_face_422', 'z_face_423', 'z_face_424', 'z_face_425', 'z_face_426', 'z_face_427', 'z_face_428', 'z_face_429', 'z_face_430', 'z_face_431', 'z_face_432', 'z_face_433', 'z_face_434', 'z_face_435', 'z_face_436', 'z_face_437', 'z_face_438', 'z_face_439', 'z_face_440', 'z_face_441', 'z_face_442', 'z_face_443', 'z_face_444', 'z_face_445', 'z_face_446', 'z_face_447', 'z_face_448', 'z_face_449', 'z_face_450', 'z_face_451', 'z_face_452', 'z_face_453', 'z_face_454', 'z_face_455', 'z_face_456', 'z_face_457', 'z_face_458', 'z_face_459', 'z_face_460', 'z_face_461', 'z_face_462', 'z_face_463', 'z_face_464', 'z_face_465', 'z_face_466', 'z_face_467', 'z_left_hand_0', 'z_left_hand_1', 'z_left_hand_2', 'z_left_hand_3', 'z_left_hand_4', 'z_left_hand_5', 'z_left_hand_6', 'z_left_hand_7', 'z_left_hand_8', 'z_left_hand_9', 'z_left_hand_10', 'z_left_hand_11', 'z_left_hand_12', 'z_left_hand_13', 'z_left_hand_14', 'z_left_hand_15', 'z_left_hand_16', 'z_left_hand_17', 'z_left_hand_18', 'z_left_hand_19', 'z_left_hand_20', 'z_pose_0', 'z_pose_1', 'z_pose_2', 'z_pose_3', 'z_pose_4', 'z_pose_5', 'z_pose_6', 'z_pose_7', 'z_pose_8', 'z_pose_9', 'z_pose_10', 'z_pose_11', 'z_pose_12', 'z_pose_13', 'z_pose_14', 'z_pose_15', 'z_pose_16', 'z_pose_17', 'z_pose_18', 'z_pose_19', 'z_pose_20', 'z_pose_21', 'z_pose_22', 'z_pose_23', 'z_pose_24', 'z_pose_25', 'z_pose_26', 'z_pose_27', 'z_pose_28', 'z_pose_29', 'z_pose_30', 'z_pose_31', 'z_pose_32', 'z_right_hand_0', 'z_right_hand_1', 'z_right_hand_2', 'z_right_hand_3', 'z_right_hand_4', 'z_right_hand_5', 'z_right_hand_6', 'z_right_hand_7', 'z_right_hand_8', 'z_right_hand_9', 'z_right_hand_10', 'z_right_hand_11', 'z_right_hand_12', 'z_right_hand_13', 'z_right_hand_14', 'z_right_hand_15', 'z_right_hand_16', 'z_right_hand_17', 'z_right_hand_18', 'z_right_hand_19', 'z_right_hand_20'])
    if filter_columns:
        FG = FeatureGenerator()
        columns = columns.reshape(3, -1)[:FG.num_axes, FG.point_arr].flatten()
    return columns


def get_seqs(seq_ids, filter_columns=True):
    FG = FeatureGenerator()
    columns = get_column_names(filter_columns)
    partial_paths = get_meta().filter(pl.col('sequence_id').is_in(seq_ids)).select('path')\
        .unique().collect().to_numpy().flatten().tolist()
    paths = ['raw_data/' + path for path in partial_paths]
    file_seq_list = []
    file_seq_ids = []
    for path in (pbar := tqdm(paths, file=sys.stdout)):
        pbar.set_description('getting sequences')
        df = pl.scan_parquet(path).filter(pl.col('sequence_id').is_in(seq_ids))
        df_seq_ids = df.select('sequence_id').collect().to_numpy().flatten()
        split_idxs = np.argwhere(df_seq_ids[1:] != df_seq_ids[:-1]).flatten() + 1
        seqs = df.select(columns).collect().to_numpy().reshape((-1, FG.num_axes, len(columns) // FG.num_axes))\
            .transpose(0, 2, 1).astype(np.float16)
        seqs = np.where(np.isnan(seqs), np.zeros_like(seqs), seqs)
        seqs = np.split(seqs, split_idxs)
        for seq in seqs:
            file_seq_list.append(seq[:FG.max_len])
        if len(split_idxs) == 0:
            unique_seq_ids = df_seq_ids[0].reshape((1,))
        else:
            unique_seq_ids = df_seq_ids[np.concatenate([np.array([0]), split_idxs])]
        file_seq_ids.append(unique_seq_ids)
    file_seq_ids = np.concatenate(file_seq_ids)
    argsort_idxs = np.argsort(seq_ids, axis=0)
    file_argsort_idxs = np.argsort(file_seq_ids, axis=0)
    inv_argsort_idxs = np.empty_like(argsort_idxs)
    inv_argsort_idxs[argsort_idxs] = np.arange(len(argsort_idxs))
    perm_idxs = file_argsort_idxs[inv_argsort_idxs]
    seq_list = []
    for idx in perm_idxs:
        seq_list.append(file_seq_list[idx])
    return seq_list


class NPZDataset(Dataset):
    @staticmethod
    def create(seq_ids, save_path, crop_labels):
        seqs = get_seqs(seq_ids)
        labels = phrases_to_labels(get_phrases(seq_ids))
        x_list, y_list = [[], []]
        xlen_list, ylen_list = np.empty(len(seq_ids), dtype=np.int32), np.empty(len(seq_ids), dtype=np.int32)
        for i, (x, y) in enumerate(pbar := tqdm(list(zip(seqs, labels)), file=sys.stdout)):
            pbar.set_description(f'creating npz dataset {save_path}')
            x_list.append(np.array(x))
            y = np.array(y, dtype=np.int32)
            if crop_labels:
                con_idxs = np.argwhere(y[1:] == y[:-1]).flatten() + 1
                if len(con_idxs) != 0:
                    y = np.insert(y, con_idxs, np.zeros_like(con_idxs))
                y = y[:len(x)]  # ensures that loss won't be nan TODO try center crop/smarter scheme
                y = y[y != 0]  # ctc loss targets can't contain the blank index
            y_list.append(y)
            xlen_list[i] = len(x)
            ylen_list[i] = len(y)
        x_list = np.concatenate(x_list)
        y_list = np.concatenate(y_list)
        save_arrs([x_list, y_list, xlen_list, ylen_list], save_path)

    def __init__(self, save_path):
        x_list, y_list, xlen_list, ylen_list = load_arrs(save_path)
        self.x_list = torch.split(torch.from_numpy(x_list), xlen_list.tolist())
        self.y_list = torch.split(torch.from_numpy(y_list), ylen_list.tolist())
        self.xlen_list = torch.from_numpy(xlen_list)
        self.ylen_list = torch.from_numpy(ylen_list)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx], self.xlen_list[idx], self.ylen_list[idx]


def get_dataloader(save_path, batch_size, shuffle):
    FG = FeatureGenerator()
    dataset = NPZDataset(save_path=save_path)
    len_counts = np.zeros(FG.max_len + 1)
    for _, _, xlen, _ in dataset:
        len_counts[xlen] += 1

    @functools.lru_cache(maxsize=None)
    def get_max_sizes(curr_len, chunks):
        if curr_len == 0:
            return 0, []
        if len_counts[curr_len] == 0:
            return get_max_sizes(curr_len - 1, chunks)
        if chunks == 1:
            return curr_len * np.sum(len_counts[:curr_len + 1]), [curr_len]
        curr_sum = 0
        min_res, min_seq = None, None
        for last_len in range(curr_len, 0, -1):
            curr_sum += curr_len * len_counts[last_len]
            next_sum, seq = get_max_sizes(last_len - 1, chunks - 1)
            seq = deepcopy(seq)
            curr_res = curr_sum + next_sum
            seq.append(curr_len)
            if min_res is None or curr_res < min_res:
                min_res = curr_res
                min_seq = seq
        return min_res, min_seq

    num_chunks = 4
    sys.setrecursionlimit(10000)
    min_len_sum, max_sizes = get_max_sizes(FG.max_len, num_chunks)
    max_sizes.pop()
    print('max_sizes:', max_sizes)
    print('estimated mean padded sample len:', min_len_sum / len(dataset))

    def collate_fn(batch):
        chunks = [([], [], [], []) for _ in range(len(max_sizes) + 1)]
        for x, y, xlen, ylen in batch:
            chunk_idx = np.searchsorted(max_sizes, xlen)
            chunks[chunk_idx][0].append(x)
            chunks[chunk_idx][1].append(y)
            chunks[chunk_idx][2].append(xlen)
            chunks[chunk_idx][3].append(ylen)
        padded_chunks = []
        for i in range(len(chunks)):
            x, y, xlen, ylen = chunks[i]
            if len(x) == 0:
                continue
            x = pad_sequence(x, batch_first=True)
            y = torch.cat(y)
            xlen = torch.stack(xlen)
            ylen = torch.stack(ylen)
            padded_chunks.append((x, y, xlen, ylen))
        return padded_chunks

    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)