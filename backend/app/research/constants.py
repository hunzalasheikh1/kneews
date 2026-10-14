FINDINGS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
            'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
STATES = ('present', 'absent', 'uncertain', 'unmentioned')
STATE_INDEX = {name: i for i, name in enumerate(STATES)}
