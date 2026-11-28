"""Request-level Bloomberg failures must never become empty market data."""
import pytest
from bbg_fetch import _blp_api as api


class Message:
    def __init__(self, values):
        self.values=values
    def hasElement(self, name):
        return name in self.values
    def getElement(self, name):
        return self.values[name]


class Event:
    def __init__(self, kind, messages=()):
        self.kind,self.messages=kind,messages
    def eventType(self):
        return self.kind
    def __iter__(self):
        return iter(self.messages)


class Session:
    def __init__(self, events):
        self.events=iter(events)
    def nextEvent(self, timeout):
        return next(self.events)


@pytest.mark.parametrize('kind',[api.blpapi.Event.RESPONSE,api.blpapi.Event.PARTIAL_RESPONSE])
def test_request_error_raises_instead_of_returning_no_data(kind):
    failed=Message({'responseError':{'category':'LIMIT','subcategory':'DAILY_CAPACITY_REACHED',
                                    'message':'Daily capacity reached.'}})
    session=Session([Event(kind,[failed]),Event(api.blpapi.Event.RESPONSE)])
    with pytest.raises(RuntimeError,match='DAILY_CAPACITY_REACHED') as info:
        api._collect_responses(session)
    assert info.value.partial_messages==[]


def test_failed_final_response_does_not_promote_partial_history():
    partial=Message({'securityData':{'security':'HYG US Equity'}})
    failed=Message({'responseError':{'category':'NOT_AUTHORIZED','message':'Access denied'}})
    session=Session([Event(api.blpapi.Event.PARTIAL_RESPONSE,[partial]),
                     Event(api.blpapi.Event.RESPONSE,[failed])])
    with pytest.raises(RuntimeError,match='Access denied') as info:
        api._collect_responses(session)
    assert info.value.partial_messages==[partial]


def test_successful_partial_and_final_payloads_are_preserved():
    a,b=Message({'securityData':1}),Message({'securityData':2})
    session=Session([Event(api.blpapi.Event.ADMIN),Event(api.blpapi.Event.PARTIAL_RESPONSE,[a]),
                     Event(api.blpapi.Event.RESPONSE,[b])])
    assert api._collect_responses(session)==[a,b]


def test_empty_success_is_distinct_from_request_error():
    assert api._collect_responses(Session([Event(api.blpapi.Event.RESPONSE)]))==[]


def test_timeout_keeps_partial_evidence():
    data=Message({'securityData':1})
    session=Session([Event(api.blpapi.Event.PARTIAL_RESPONSE,[data]),Event(api.blpapi.Event.TIMEOUT)])
    with pytest.raises(TimeoutError) as info:
        api._collect_responses(session,timeout_ms=5)
    assert info.value.partial_messages==[data]
