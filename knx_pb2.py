# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: knx.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import atlas_pb2 as atlas__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='knx.proto',
  package='',
  syntax='proto3',
  serialized_options=_b('\n\026org.atlas.messages.knxB\rKnxDescriptorP\001'),
  serialized_pb=_b('\n\tknx.proto\x1a\x0b\x61tlas.proto\x1a\x1fgoogle/protobuf/timestamp.proto\")\n\x0bScadaObject\x12\r\n\x05label\x18\x01 \x01(\t\x12\x0b\n\x03url\x18\x02 \x01(\t\"\x80\x01\n\x0eKnxScadaAction\x12#\n\x06header\x18\x01 \x01(\x0b\x32\x13.AtlasMessageHeader\x12\x1c\n\x06object\x18\x02 \x01(\x0b\x32\x0c.ScadaObject\x12\x1c\n\x06\x61\x63tion\x18\x03 \x01(\x0e\x32\x0c.ScadaAction\x12\r\n\x05value\x18\x04 \x01(\t*G\n\x0bScadaAction\x12\x15\n\x11SCADA_ACTION_NONE\x10\x00\x12\t\n\x05WRITE\x10\x01\x12\x08\n\x04READ\x10\x02\x12\x0c\n\x08GETVALUE\x10\x03\x42)\n\x16org.atlas.messages.knxB\rKnxDescriptorP\x01\x62\x06proto3')
  ,
  dependencies=[atlas__pb2.DESCRIPTOR,google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,])

_SCADAACTION = _descriptor.EnumDescriptor(
  name='ScadaAction',
  full_name='ScadaAction',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SCADA_ACTION_NONE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WRITE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='READ', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GETVALUE', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=233,
  serialized_end=304,
)
_sym_db.RegisterEnumDescriptor(_SCADAACTION)

ScadaAction = enum_type_wrapper.EnumTypeWrapper(_SCADAACTION)
SCADA_ACTION_NONE = 0
WRITE = 1
READ = 2
GETVALUE = 3



_SCADAOBJECT = _descriptor.Descriptor(
  name='ScadaObject',
  full_name='ScadaObject',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='label', full_name='ScadaObject.label', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='url', full_name='ScadaObject.url', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=59,
  serialized_end=100,
)


_KNXSCADAACTION = _descriptor.Descriptor(
  name='KnxScadaAction',
  full_name='KnxScadaAction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='KnxScadaAction.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='object', full_name='KnxScadaAction.object', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action', full_name='KnxScadaAction.action', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='KnxScadaAction.value', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=103,
  serialized_end=231,
)

_KNXSCADAACTION.fields_by_name['header'].message_type = atlas__pb2._ATLASMESSAGEHEADER
_KNXSCADAACTION.fields_by_name['object'].message_type = _SCADAOBJECT
_KNXSCADAACTION.fields_by_name['action'].enum_type = _SCADAACTION
DESCRIPTOR.message_types_by_name['ScadaObject'] = _SCADAOBJECT
DESCRIPTOR.message_types_by_name['KnxScadaAction'] = _KNXSCADAACTION
DESCRIPTOR.enum_types_by_name['ScadaAction'] = _SCADAACTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ScadaObject = _reflection.GeneratedProtocolMessageType('ScadaObject', (_message.Message,), dict(
  DESCRIPTOR = _SCADAOBJECT,
  __module__ = 'knx_pb2'
  # @@protoc_insertion_point(class_scope:ScadaObject)
  ))
_sym_db.RegisterMessage(ScadaObject)

KnxScadaAction = _reflection.GeneratedProtocolMessageType('KnxScadaAction', (_message.Message,), dict(
  DESCRIPTOR = _KNXSCADAACTION,
  __module__ = 'knx_pb2'
  # @@protoc_insertion_point(class_scope:KnxScadaAction)
  ))
_sym_db.RegisterMessage(KnxScadaAction)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
