"""
Definition of the OPC UA Server
"""

from opcua.ua import NodeId, NodeIdType
from opcua import Server, ua
import pickle

#  Definition of the pickle file
# from datetime import datetime
# with open('pickles/valve01.p', 'wb') as handle:
#     pickle.dump({'typ':'Diaphragm Valve', 
#                  'medium':'air', 
#                  'status':'open', 
#                  'accuracy':'0.9', 
#                  'updated':datetime(2020, 4, 25, 10, 26, 33, 187313), 
#                  'cycles':25}, handle, protocol=pickle.HIGHEST_PROTOCOL)

class ServerClass:
    def define_server(self):
        """
        Function for setting up the OPC UA Server.

        """
        #  Instanciation of OPC-UA Server
        self.server = Server()        
        self.server.set_endpoint("opc.tcp://192.168.178.44:4840/")
        self.server.set_server_name('Opc UA Valve-Sensor Server')
        # set all possible endpoint policies for clients to connect through
        self.server.set_security_policy([
                    ua.SecurityPolicyType.NoSecurity,
                    ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt,
                    ua.SecurityPolicyType.Basic256Sha256_Sign])       
        
        # setup our own namespace
        self.uri = "http://test.valve.iit"
        self.idx = self.server.register_namespace(self.uri)
        
        # Create a new node type we can instantiate in address space
        self.dev = self.server.nodes.base_object_type.add_object_type(self.idx,
                                                                      "Test Server")
        
        self.devices = self.server.nodes.objects.add_object(self.idx,
                                                            "Devices")
        self.valve01 = self.devices.add_object(self.idx,
                                               'Valve01')

        
        #  Definition of the pickle file
        with open('pickles/valve01.p', 'rb') as handle:
            self.valve_prop = pickle.load(handle)
        
        #  Definition of the OPC UA classes and variables
        self.typ_valve01 = self.valve01.add_variable(self.idx,
                                                     "Valve Type",
                                                     self.valve_prop['typ'])
        self.medium_valve01 = self.valve01.add_variable(self.idx,
                                                        "Medium",
                                                        self.valve_prop['medium'])
        
        self.status_valve01 = self.valve01.add_variable(self.idx,
                                                        "Status",
                                                        self.valve_prop['status'])
        self.status_valve01.set_writable()
        
        self.accuracy_valve01 = self.valve01.add_variable(self.idx, "Accuracy",
                                                          self.valve_prop['accuracy'],
                                                          ua.VariantType.Float)
        self.accuracy_valve01.set_writable()
        
        self.status_update_valve01 = self.valve01.add_variable(self.idx,
                                                               "Last Status Update",
                                                               self.valve_prop['updated'])
        self.status_update_valve01.set_writable()  
        
        self.cycles_valve01 = self.valve01.add_variable(self.idx,
                                                        'Total Cycles',
                                                        self.valve_prop['cycles'])
        self.cycles_valve01.set_writable() 
        
        self.server.start()
    
    def use_server(self, status, updated, accuracy, cycles):
        """
        Function for updating the OPC UA Server.

        """
        self.status_valve01.set_value(status)
        self.status_update_valve01.set_value(updated)
        self.accuracy_valve01.set_value(accuracy)
        self.cycles_valve01.set_value(cycles)
    
    def stop_server(self):
        """
        Function for stopping the OPC UA Server.

        """
        self.server.stop()