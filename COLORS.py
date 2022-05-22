class COLORS:
    """
    The xqcd color survey in BGR format
    
    Use example:
    colors = COLORS()
    color = colors.next
    """
    @property
    def next(self):
        color = self.colors[self.index]
        self.index += 1
        if self.index >= len(self.colors): self.index = 0
        return color

    def __init__(self):
        self.index = 0
        self.colors = [(217, 194, 172),
        (87, 174, 86),
        (110, 153, 178),
        (4, 255, 168),
        (79, 216, 105),
        (133, 69, 137),
        (63, 178, 112),
        (255, 255, 212),
        (124, 171, 101),
        (143, 46, 149),
        (129, 252, 252),
        (145, 163, 165),
        (4, 128, 56),
        (133, 144, 76),
        (138, 155, 94),
        (53, 180, 239),
        (130, 155, 217),
        (56, 95, 10),
        (247, 6, 12),
        (42, 222, 97),
        (191, 120, 55),
        (199, 66, 34),
        (198, 60, 83),
        (60, 181, 155),
        (166, 255, 5),
        (87, 99, 31),
        (116, 115, 1),
        (119, 181, 12),
        (137, 7, 255),
        (139, 168, 175),
        (127, 120, 8),
        (215, 133, 221),
        (117, 200, 166),
        (181, 255, 167),
        (9, 183, 194),
        (165, 142, 231),
        (189, 110, 150),
        (96, 173, 204),
        (168, 134, 172),
        (148, 126, 148),
        (178, 63, 152),
        (233, 99, 255),
        (165, 251, 178),
        (101, 179, 99),
        (63, 229, 142),
        (161, 225, 183),
        (82, 111, 255),
        (163, 248, 189),
        (131, 182, 211),
        (196, 252, 255),
        (65, 5, 67),
        (208, 178, 255),
        (112, 117, 153),
        (13, 144, 173),
        (253, 142, 196),
        (156, 123, 80),
        (3, 113, 125),
        (120, 253, 255),
        (125, 70, 218),
        (0, 2, 65),
        (121, 209, 201),
        (134, 250, 255),
        (174, 132, 86),
        (133, 124, 107),
        (10, 108, 111),
        (113, 64, 126),
        (55, 147, 0),
        (41, 228, 208),
        (23, 249, 255),
        (236, 93, 29),
        (7, 73, 5),
        (8, 206, 181),
        (123, 182, 143),
        (176, 255, 200),
        (108, 222, 253),
        (34, 223, 255),
        (112, 190, 169),
        (227, 50, 104),
        (71, 177, 253),
        (125, 172, 199),
        (154, 243, 255),
        (4, 14, 133),
        (254, 192, 239),
        (20, 253, 64),
        (6, 196, 182),
        (0, 255, 157),
        (66, 65, 60),
        (21, 171, 242),
        (6, 79, 172),
        (130, 254, 196),
        (31, 250, 44),
        (0, 98, 154),
        (247, 155, 202),
        (66, 95, 135),
        (254, 46, 58),
        (73, 141, 253),
        (3, 49, 139),
        (96, 165, 203),
        (57, 131, 105),
        (115, 220, 12),
        (3, 82, 183),
        (78, 143, 127),
        (141, 83, 38),
        (80, 169, 99),
        (137, 127, 200),
        (153, 252, 177),
        (138, 154, 255),
        (142, 104, 246),
        (168, 253, 118),
        (92, 254, 83),
        (84, 253, 78),
        (191, 254, 160),
        (218, 242, 123),
        (166, 245, 188),
        (2, 107, 202),
        (176, 122, 16),
        (171, 56, 33),
        (145, 159, 113),
        (21, 185, 253),
        (175, 252, 254),
        (121, 246, 252),
        (0, 2, 29),
        (67, 104, 203),
        (138, 102, 49),
        (253, 122, 36),
        (182, 255, 255),
        (169, 253, 144),
        (125, 161, 134),
        (92, 220, 253),
        (182, 209, 120),
        (175, 187, 19),
        (252, 95, 251),
        (134, 249, 32),
        (110, 227, 255),
        (89, 7, 157),
        (177, 24, 58),
        (137, 255, 194),
        (173, 103, 215),
        (88, 0, 114),
        (3, 218, 255),
        (141, 192, 1),
        (52, 116, 172),
        (0, 70, 1),
        (250, 0, 153),
        (111, 6, 2),
        (24, 118, 142),
        (143, 118, 209),
        (3, 180, 150),
        (99, 255, 253),
        (166, 163, 149),
        (78, 104, 127),
        (115, 25, 117),
        (4, 148, 8),
        (99, 97, 255),
        (86, 133, 89),
        (97, 71, 33),
        (168, 115, 60),
        (136, 158, 186),
        (249, 27, 2),
        (101, 74, 115),
        (139, 196, 35),
        (34, 174, 143),
        (162, 242, 230),
        (219, 87, 75),
        (102, 1, 217),
        (130, 84, 1),
        (22, 2, 157),
        (2, 143, 114),
        (173, 229, 255),
        (80, 5, 78),
        (8, 188, 249),
        (58, 7, 255),
        (134, 121, 199),
        (254, 255, 214),
        (3, 75, 254),
        (86, 89, 253),
        (102, 225, 252),
        (61, 113, 178),
        (77, 59, 31),
        (76, 157, 105),
        (162, 252, 86),
        (129, 85, 251),
        (252, 130, 62),
        (22, 191, 160),
        (250, 255, 214),
        (142, 115, 79),
        (154, 177, 255),
        (21, 139, 92),
        (104, 172, 84),
        (176, 160, 137),
        (122, 160, 126),
        (6, 252, 27),
        (251, 255, 202),
        (187, 255, 182),
        (9, 94, 167),
        (255, 46, 21),
        (183, 94, 141),
        (143, 158, 95),
        (180, 247, 99),
        (2, 102, 96),
        (170, 134, 252),
        (52, 0, 140),
        (0, 128, 117),
        (76, 126, 171),
        (100, 7, 3),
        (164, 134, 254),
        (78, 23, 213),
        (252, 208, 254),
        (24, 0, 104),
        (8, 223, 254),
        (15, 66, 254),
        (0, 124, 111),
        (71, 1, 202),
        (49, 36, 27),
        (176, 251, 0),
        (86, 88, 219),
        (24, 214, 221),
        (254, 253, 65),
        (78, 82, 207),
        (111, 195, 33),
        (8, 3, 169),
        (5, 16, 110),
        (140, 130, 254),
        (19, 97, 75),
        (9, 164, 77),
        (138, 174, 190),
        (248, 57, 3),
        (89, 143, 168),
        (208, 33, 93),
        (9, 178, 254),
        (139, 81, 78),
        (2, 78, 150),
        (178, 163, 133),
        (175, 105, 255),
        (244, 251, 195),
        (183, 254, 42),
        (106, 95, 0),
        (147, 23, 12),
        (129, 255, 255),
        (58, 131, 240),
        (63, 243, 241),
        (123, 210, 177),
        (74, 130, 252),
        (52, 170, 113),
        (226, 201, 183),
        (1, 1, 75),
        (230, 82, 165),
        (13, 47, 175),
        (248, 136, 139),
        (100, 247, 154),
        (178, 251, 166),
        (18, 197, 255),
        (81, 8, 117),
        (9, 74, 193),
        (74, 47, 254),
        (226, 3, 2),
        (122, 67, 10),
        (85, 0, 165),
        (12, 139, 174),
        (143, 121, 253),
        (5, 172, 191),
        (118, 175, 62),
        (103, 71, 199),
        (78, 72, 185),
        (142, 125, 100),
        (40, 254, 191),
        (222, 37, 215),
        (5, 151, 178),
        (63, 58, 103),
        (194, 125, 168),
        (75, 254, 250),
        (47, 2, 192),
        (204, 135, 14),
        (104, 132, 141),
        (222, 3, 173),
        (158, 255, 140),
        (2, 172, 148),
        (247, 255, 196),
        (115, 238, 253),
        (100, 184, 51),
        (208, 249, 255),
        (163, 141, 117),
        (201, 4, 245),
        (181, 161, 119),
        (228, 86, 135),
        (23, 151, 136),
        (121, 126, 194),
        (113, 115, 1),
        (3, 131, 159),
        (96, 213, 247),
        (254, 246, 189),
        (79, 184, 117),
        (4, 187, 156),
        (91, 70, 41),
        (6, 96, 105),
        (2, 248, 173),
        (252, 198, 193),
        (107, 173, 53),
        (55, 253, 255),
        (160, 66, 164),
        (150, 97, 243),
        (6, 119, 148),
        (242, 244, 255),
        (103, 145, 30),
        (6, 195, 181),
        (127, 255, 254),
        (188, 253, 207),
        (8, 221, 10),
        (5, 253, 135),
        (118, 248, 30),
        (199, 253, 123),
        (172, 236, 188),
        (15, 249, 187),
        (4, 144, 171),
        (122, 181, 31),
        (90, 85, 0),
        (172, 132, 164),
        (8, 85, 196),
        (157, 130, 63),
        (68, 141, 84),
        (251, 94, 201),
        (127, 229, 58),
        (149, 103, 1),
        (34, 169, 135),
        (77, 148, 240),
        (81, 20, 93),
        (41, 255, 37),
        (29, 254, 208),
        (43, 166, 255),
        (76, 180, 1),
        (181, 108, 255),
        (71, 66, 107),
        (12, 193, 199),
        (250, 255, 183),
        (110, 255, 174),
        (1, 45, 236),
        (123, 255, 118),
        (57, 0, 115),
        (72, 3, 4),
        (200, 78, 223),
        (60, 203, 110),
        (5, 152, 143),
        (31, 220, 94),
        (245, 79, 217),
        (61, 253, 200),
        (13, 13, 7),
        (184, 132, 73),
        (59, 183, 81),
        (4, 126, 172),
        (129, 84, 78),
        (75, 110, 135),
        (8, 188, 88),
        (16, 239, 47),
        (84, 254, 45),
        (2, 255, 10),
        (67, 239, 156),
        (123, 209, 24),
        (10, 83, 53),
        (219, 5, 24),
        (196, 88, 98),
        (79, 150, 255),
        (15, 171, 255),
        (231, 140, 143),
        (168, 188, 36),
        (44, 1, 63),
        (95, 248, 203),
        (76, 114, 255),
        (55, 1, 40),
        (246, 111, 179),
        (114, 192, 72),
        (122, 203, 188),
        (91, 65, 168),
        (196, 177, 6),
        (132, 117, 205),
        (122, 218, 241),
        (144, 4, 255),
        (135, 91, 128),
        (71, 167, 80),
        (149, 164, 168),
        (4, 255, 207),
        (126, 255, 255),
        (167, 127, 255),
        (38, 64, 239),
        (146, 153, 60),
        (6, 104, 136),
        (137, 244, 4),
        (158, 246, 254),
        (123, 175, 207),
        (159, 113, 59),
        (197, 193, 253),
        (115, 192, 32),
        (192, 95, 155),
        (142, 155, 15),
        (2, 40, 116),
        (44, 185, 157),
        (32, 191, 164),
        (9, 89, 205),
        (135, 165, 173),
        (60, 1, 190),
        (235, 255, 184),
        (1, 77, 220),
        (62, 101, 162),
        (39, 139, 99),
        (3, 156, 65),
        (101, 255, 177),
        (212, 188, 157),
        (254, 253, 253),
        (86, 171, 119),
        (150, 65, 70),
        (71, 1, 153),
        (115, 253, 190),
        (132, 191, 50),
        (9, 111, 175),
        (92, 2, 160),
        (177, 216, 255),
        (30, 78, 127),
        (12, 155, 191),
        (83, 163, 107),
        (230, 117, 240),
        (246, 200, 123),
        (148, 95, 71),
        (3, 191, 245),
        (182, 254, 255),
        (116, 253, 255),
        (123, 91, 137),
        (173, 107, 67),
        (1, 193, 208),
        (8, 248, 198),
        (5, 54, 244),
        (77, 193, 2),
        (3, 95, 178),
        (25, 126, 42),
        (72, 6, 73),
        (103, 98, 83),
        (239, 6, 90),
        (52, 2, 207),
        (97, 166, 196),
        (132, 138, 151),
        (84, 9, 31),
        (45, 1, 3),
        (121, 177, 43),
        (155, 144, 195),
        (181, 111, 166),
        (1, 0, 119),
        (5, 43, 146),
        (124, 127, 125),
        (75, 15, 153),
        (3, 115, 143),
        (185, 60, 200),
        (147, 169, 254),
        (13, 187, 172),
        (254, 113, 192),
        (127, 253, 204),
        (46, 2, 0),
        (68, 131, 130),
        (203, 197, 255),
        (57, 18, 171),
        (75, 5, 176),
        (4, 204, 153),
        (0, 124, 147),
        (41, 149, 1),
        (231, 29, 239),
        (53, 4, 0),
        (149, 179, 66),
        (131, 87, 157),
        (169, 172, 200),
        (6, 118, 200),
        (4, 39, 170),
        (255, 203, 228),
        (36, 66, 250),
        (249, 4, 8),
        (0, 178, 92),
        (78, 66, 118),
        (14, 122, 108),
        (126, 221, 251),
        (52, 1, 42),
        (5, 74, 4),
        (89, 70, 253),
        (248, 117, 13),
        (2, 0, 254),
        (6, 157, 203),
        (7, 125, 251),
        (129, 204, 185),
        (255, 200, 237),
        (96, 225, 97),
        (254, 184, 138),
        (78, 10, 146),
        (162, 2, 254),
        (1, 48, 154),
        (8, 254, 101),
        (183, 253, 190),
        (97, 114, 177),
        (1, 95, 136),
        (254, 204, 2),
        (149, 253, 193),
        (57, 101, 131),
        (67, 41, 251),
        (1, 183, 132),
        (37, 99, 182),
        (18, 81, 127),
        (82, 160, 95),
        (253, 237, 109),
        (234, 249, 11),
        (255, 96, 199),
        (203, 255, 255),
        (252, 206, 246),
        (132, 80, 21),
        (79, 5, 245),
        (3, 84, 100),
        (1, 89, 122),
        (4, 181, 168),
        (115, 153, 61),
        (51, 1, 0),
        (115, 169, 118),
        (136, 90, 46),
        (125, 247, 11),
        (72, 108, 189),
        (184, 29, 172),
        (106, 175, 43),
        (253, 247, 38),
        (108, 253, 174),
        (85, 143, 155),
        (1, 173, 255),
        (4, 156, 198),
        (84, 208, 244),
        (172, 157, 222),
        (13, 72, 5),
        (116, 174, 201),
        (15, 70, 96),
        (176, 246, 152),
        (254, 241, 138),
        (187, 232, 46),
        (93, 135, 17),
        (192, 176, 253),
        (2, 96, 177),
        (42, 2, 247),
        (9, 171, 213),
        (95, 119, 134),
        (89, 159, 198),
        (127, 104, 122),
        (96, 46, 4),
        (148, 141, 200),
        (213, 251, 165),
        (113, 254, 255),
        (199, 65, 98),
        (64, 254, 255),
        (78, 73, 211),
        (43, 94, 152),
        (76, 129, 166),
        (232, 8, 255),
        (81, 118, 157),
        (202, 255, 254),
        (141, 86, 152),
        (58, 0, 158),
        (55, 124, 40),
        (2, 105, 185),
        (115, 104, 186),
        (85, 120, 255),
        (28, 178, 148),
        (199, 201, 197),
        (238, 26, 102),
        (239, 64, 97),
        (170, 229, 155),
        (4, 88, 123),
        (179, 106, 39),
        (8, 179, 254),
        (126, 253, 140),
        (234, 136, 100),
        (238, 110, 5),
        (1, 122, 178),
        (249, 254, 15),
        (85, 42, 250),
        (71, 7, 130),
        (79, 106, 122),
        (12, 50, 244),
        (5, 57, 161),
        (138, 130, 111),
        (244, 90, 165),
        (253, 10, 173),
        (119, 69, 0),
        (109, 141, 101),
        (128, 123, 202),
        (73, 82, 0),
        (52, 93, 43),
        (40, 241, 191),
        (16, 148, 181),
        (187, 118, 41),
        (130, 65, 1),
        (63, 63, 187),
        (71, 38, 252),
        (0, 121, 168),
        (178, 203, 130),
        (62, 124, 102),
        (165, 70, 254),
        (204, 131, 254),
        (23, 166, 148),
        (5, 137, 168),
        (0, 95, 127),
        (162, 67, 158),
        (3, 46, 6),
        (69, 110, 138),
        (139, 122, 204),
        (104, 1, 158),
        (56, 255, 253),
        (139, 250, 192),
        (91, 220, 238),
        (1, 189, 126),
        (146, 91, 59),
        (159, 136, 1),
        (253, 122, 61),
        (231, 52, 95),
        (207, 90, 109),
        (0, 133, 116),
        (17, 108, 112),
        (8, 0, 60),
        (245, 0, 203),
        (4, 45, 0),
        (187, 140, 101),
        (81, 149, 116),
        (102, 255, 185),
        (0, 193, 157),
        (102, 238, 250),
        (179, 251, 126),
        (44, 0, 123),
        (161, 146, 194),
        (146, 123, 1),
        (6, 192, 252),
        (50, 116, 101),
        (59, 134, 216),
        (149, 133, 115),
        (255, 35, 170),
        (8, 255, 8),
        (1, 122, 155),
        (142, 158, 242),
        (118, 194, 111),
        (0, 91, 255),
        (82, 255, 253),
        (133, 111, 134),
        (9, 254, 143),
        (254, 207, 238),
        (201, 10, 81),
        (83, 145, 79),
        (5, 35, 159),
        (57, 134, 114),
        (98, 12, 222),
        (153, 110, 145),
        (109, 177, 255),
        (3, 77, 60),
        (83, 112, 127),
        (111, 146, 119),
        (204, 15, 1),
        (250, 174, 206),
        (251, 153, 143),
        (255, 252, 198),
        (204, 57, 85),
        (3, 78, 84),
        (121, 122, 1),
        (198, 249, 1),
        (3, 176, 201),
        (1, 153, 146),
        (9, 85, 11),
        (152, 4, 160),
        (177, 0, 32),
        (140, 86, 148),
        (14, 190, 194),
        (151, 139, 116),
        (209, 95, 102),
        (165, 109, 156),
        (64, 66, 196),
        (87, 72, 162),
        (135, 95, 130),
        (59, 100, 201),
        (52, 177, 144),
        (106, 56, 1),
        (111, 163, 37),
        (109, 101, 89),
        (99, 253, 117),
        (13, 252, 33),
        (173, 134, 90),
        (21, 198, 254),
        (1, 253, 255),
        (254, 197, 223),
        (0, 100, 178),
        (0, 94, 127),
        (93, 126, 222),
        (67, 130, 4),
        (212, 255, 255),
        (140, 99, 59),
        (0, 148, 183),
        (126, 89, 132),
        (0, 25, 65),
        (35, 3, 123),
        (255, 217, 4),
        (44, 126, 102),
        (172, 238, 251),
        (254, 255, 215),
        (150, 116, 78),
        (98, 76, 135),
        (255, 255, 213),
        (140, 109, 130),
        (205, 186, 255),
        (189, 255, 209),
        (228, 142, 68),
        (42, 71, 5),
        (157, 134, 213),
        (52, 7, 61),
        (0, 1, 74),
        (28, 72, 248),
        (15, 89, 2),
        (3, 162, 137),
        (216, 63, 224),
        (148, 138, 213),
        (116, 178, 123),
        (37, 101, 82),
        (190, 76, 201),
        (218, 75, 219),
        (35, 54, 158),
        (93, 72, 181),
        (18, 92, 115),
        (87, 109, 156),
        (30, 143, 2),
        (110, 145, 177),
        (156, 117, 73),
        (14, 69, 160),
        (72, 173, 57),
        (80, 106, 182),
        (219, 255, 140),
        (92, 190, 164),
        (35, 119, 203),
        (107, 105, 5),
        (174, 93, 206),
        (83, 90, 200),
        (141, 174, 150),
        (116, 167, 31),
        (3, 151, 122),
        (98, 147, 172),
        (73, 160, 1),
        (77, 84, 217),
        (247, 95, 250),
        (252, 202, 130),
        (252, 255, 172),
        (1, 176, 252),
        (81, 9, 145),
        (84, 44, 254),
        (196, 117, 200),
        (10, 197, 205),
        (30, 65, 253),
        (0, 2, 154),
        (0, 100, 190),
        (167, 10, 3),
        (154, 1, 254),
        (154, 135, 247),
        (145, 113, 136),
        (73, 1, 176),
        (147, 225, 18),
        (124, 123, 254),
        (8, 148, 255),
        (9, 110, 106),
        (22, 46, 139),
        (18, 97, 105),
        (1, 119, 225),
        (30, 72, 10),
        (55, 56, 52),
        (206, 183, 255),
        (247, 121, 106),
        (233, 6, 93),
        (2, 28, 61),
        (125, 166, 130),
        (25, 1, 190),
        (39, 255, 201),
        (2, 62, 55),
        (30, 86, 169),
        (255, 160, 202),
        (65, 102, 202),
        (233, 216, 2),
        (120, 179, 136),
        (2, 0, 152),
        (98, 1, 203),
        (45, 172, 92),
        (88, 153, 118),
        (254, 191, 162),
        (116, 166, 16),
        (139, 180, 6),
        (74, 136, 175),
        (135, 139, 11),
        (86, 167, 255),
        (21, 164, 162),
        (6, 68, 21),
        (152, 103, 133),
        (63, 1, 52),
        (233, 45, 99),
        (138, 136, 10),
        (50, 118, 111),
        (126, 106, 212),
        (143, 72, 30),
        (254, 19, 188),
        (204, 244, 126),
        (38, 205, 118),
        (98, 166, 116),
        (63, 1, 128),
        (252, 209, 177),
        (228, 255, 255),
        (255, 82, 6),
        (90, 92, 4),
        (206, 41, 87),
        (243, 154, 6),
        (13, 0, 255),
        (69, 12, 241),
        (215, 112, 81),
        (105, 191, 172),
        (97, 52, 108),
        (157, 129, 94),
        (249, 30, 96),
        (22, 221, 176),
        (2, 253, 205),
        (187, 111, 44),
        (122, 115, 192),
        (252, 180, 214),
        (53, 0, 2),
        (231, 59, 112),
        (6, 60, 253),
        (86, 0, 150),
        (104, 163, 64),
        (156, 113, 3),
        (80, 90, 252),
        (194, 255, 255),
        (10, 43, 127),
        (15, 78, 176),
        (35, 54, 160),
        (115, 174, 135),
        (115, 155, 120),
        (255, 255, 255),
        (249, 239, 152),
        (56, 139, 101),
        (154, 125, 90),
        (53, 8, 56),
        (122, 254, 255),
        (4, 169, 92),
        (214, 220, 216),
        (2, 165, 165),
        (215, 72, 214),
        (149, 116, 4),
        (212, 144, 183),
        (153, 124, 91),
        (142, 124, 96),
        (8, 64, 11),
        (217, 13, 237),
        (15, 0, 140),
        (132, 255, 255),
        (5, 144, 191),
        (10, 189, 210),
        (76, 71, 255),
        (209, 133, 4),
        (220, 207, 255),
        (115, 2, 4),
        (9, 60, 168),
        (193, 228, 144),
        (114, 101, 81),
        (5, 194, 250),
        (10, 182, 213),
        (55, 55, 54),
        (22, 93, 75),
        (164, 139, 107),
        (173, 249, 128),
        (82, 126, 165),
        (113, 249, 169),
        (2, 81, 198),
        (118, 202, 226),
        (157, 255, 176),
        (176, 254, 159),
        (72, 170, 253),
        (177, 1, 254),
        (10, 248, 193),
        (63, 1, 54),
        (2, 28, 52),
        (129, 162, 185),
        (18, 171, 142),
        (7, 174, 154),
        (46, 171, 2),
        (171, 249, 122),
        (109, 126, 19),
        (98, 166, 170),
        (35, 0, 97),
        (78, 77, 1),
        (2, 20, 143),
        (110, 0, 75),
        (65, 15, 88),
        (159, 255, 143),
        (12, 180, 219),
        (254, 207, 162),
        (45, 251, 192),
        (253, 3, 190),
        (0, 0, 132),
        (254, 254, 208),
        (11, 155, 63),
        (62, 21, 1),
        (178, 216, 4),
        (1, 78, 192),
        (12, 255, 12),
        (252, 101, 1),
        (117, 98, 207),
        (223, 209, 255),
        (1, 179, 206),
        (130, 2, 56),
        (50, 255, 170),
        (161, 252, 83),
        (254, 130, 142),
        (107, 65, 203),
        (4, 122, 103),
        (124, 176, 255),
        (181, 253, 199),
        (80, 129, 173),
        (141, 2, 255),
        (0, 0, 0),
        (253, 162, 206),
        (70, 17, 0),
        (170, 4, 5),
        (166, 218, 230),
        (108, 121, 255),
        (14, 117, 110),
        (33, 0, 101),
        (7, 255, 1),
        (62, 6, 53),
        (129, 113, 174),
        (12, 71, 6),
        (201, 234, 19),
        (255, 255, 0),
        (111, 178, 209),
        (91, 3, 0),
        (239, 159, 199),
        (172, 194, 6),
        (0, 53, 3),
        (234, 14, 154),
        (246, 119, 191),
        (5, 254, 137),
        (145, 149, 146),
        (253, 187, 117),
        (20, 255, 255),
        (120, 0, 194),
        (123, 249, 150),
        (6, 115, 249),
        (134, 147, 2),
        (252, 208, 149),
        (0, 0, 229),
        (0, 55, 101),
        (192, 129, 255),
        (223, 67, 3),
        (26, 176, 21),
        (156, 30, 126)]

