import pytest
import copy
import json
from app.research.labelling import rule_labels,validate_labels

def test_four_states_and_no_silent_negative():
    labels=rule_labels('Complete ACL tear. MCL intact. Possible medial meniscus tear.')
    assert labels['ACL']['state']=='present'
    assert labels['MCL']['state']=='absent'
    assert labels['Medial Meniscus']['state']=='uncertain'
    assert labels['Synovitis']['state']=='unmentioned'

def test_conflicting_mentions_abstain():
    assert rule_labels('ACL intact. ACL tear.')['ACL']['state']=='uncertain'

def test_llm_evidence_must_exist_in_original_report():
    labels=rule_labels('No effusion.')
    labels['Effusion']['evidence']=['invented text']
    with pytest.raises(ValueError,match='Unsupported evidence'):
        validate_labels({'findings':labels},'No effusion.')


def model_reply(evidence_id):
    from app.research.constants import FINDINGS
    findings={name:dict(state='unmentioned',evidence_ids=[],probability=None) for name in FINDINGS}
    findings['ACL']=dict(state='present',evidence_ids=[evidence_id],probability=.9)
    return {'findings':findings}


def mock_local_model(monkeypatch,replies):
    from app.research import labelling
    requests=[]
    class Response:
        def __init__(self,payload):self.payload=payload
        def raise_for_status(self):pass
        def json(self):return {'message':{'content':json.dumps(self.payload)}}
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,url,*,json):
            requests.append(copy.deepcopy(json))
            return Response(replies[len(requests)-1])
    monkeypatch.setattr(labelling.httpx,'Client',Client)
    return requests


def test_explicit_repair_regenerates_evidence_and_records_provenance(monkeypatch):
    from app.research.labelling import label_local,LABEL_SCHEMA,EVIDENCE_REPAIR_VERSION,PROMPT
    original=copy.deepcopy(LABEL_SCHEMA)
    requests=mock_local_model(monkeypatch,[model_reply(2)]*3+[model_reply(1)])
    labels,provenance=label_local('No effusion. ACL tear.','http://127.0.0.1:11435','test:local',
                                 {'digest':'a'*64},repair_invalid_evidence=True)
    assert labels['ACL']['evidence']==[' ACL tear.']
    assert provenance['attempts']==4
    assert provenance['evidence_repair']['version']==EVIDENCE_REPAIR_VERSION
    assert len(provenance['evidence_repair']['validation_failures'])==3
    assert all(request['format']==original for request in requests[:3])
    item=requests[3]['format']['properties']['findings']['properties']['ACL']['oneOf'][0]
    assert item['properties']['evidence_ids']['items']['enum']==[0,1]
    assert requests[3]['messages'][0]['content']==PROMPT
    assert requests[3]['messages'][1]==requests[0]['messages'][1]
    assert len(requests[3]['messages'])==3
    assert LABEL_SCHEMA==original


def test_invalid_evidence_still_fails_without_opt_in(monkeypatch):
    from app.research.labelling import label_local
    requests=mock_local_model(monkeypatch,[model_reply(1)]*3)
    with pytest.raises(ValueError,match='after 3 attempts'):
        label_local('ACL tear.','http://127.0.0.1:11435','test:local',{'digest':'a'*64})
    assert len(requests)==3


@pytest.mark.parametrize('invalid_id',[1,-1,True,'0'])
def test_repair_never_clamps_shifts_or_coerces_invalid_ids(monkeypatch,invalid_id):
    from app.research.labelling import label_local
    requests=mock_local_model(monkeypatch,[model_reply(invalid_id)]*5)
    with pytest.raises(ValueError,match='after 5 attempts'):
        label_local('ACL tear.','http://127.0.0.1:11435','test:local',{'digest':'a'*64},repair_invalid_evidence=True)
    assert len(requests)==5


def test_successful_original_request_has_no_repair_marker(monkeypatch):
    from app.research.labelling import label_local,LABEL_SCHEMA
    requests=mock_local_model(monkeypatch,[model_reply(0)])
    _,provenance=label_local('ACL tear.','http://127.0.0.1:11435','test:local',{'digest':'a'*64},repair_invalid_evidence=True)
    assert provenance['attempts']==1 and 'evidence_repair' not in provenance
    assert requests[0]['format']==LABEL_SCHEMA
