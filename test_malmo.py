from malmoenv import Env
xml = open('missions/test_pvp.xml').read()
agent = Env()
agent.init(xml=xml, port=9000, server='127.0.0.1', role=0)
obs = agent.reset()    # should now work without assertion
