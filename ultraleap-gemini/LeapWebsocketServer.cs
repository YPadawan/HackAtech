using System;
using System.Collections.Generic;
using System.Reflection;
using Leap;
using UnityEngine;
using WebSocketSharp;
using WebSocketSharp.Server;
using Leap.Unity; // https://openupm.com/packages/com.ultraleap.tracking/
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

// https://docs.ultraleap.com/unity-api/The-Basics/getting-started.html
/*
 * json format of frame
 * {
 *  id
 *  timestamp
 *  currentFrameRate
 *  hands [
 *      { // hand
 *          id
 *          elbow
 *          wrist
 *          confidence
 *          grabStrength
 *          +grabAngle (???)
 *          pinchStrength
 *          pinchDistance
 *          palmWidth
 *          +type (<- isRight)
 *          timeVisible
 *          +armWidth (<- arm.width)
 *          +armBasis (<- arm.basis)
 *          palmPosition
 *          palmVelocity
 *          palmNormal
 *          direction
 *      }
 *  ]
 *  pointables [ // fingers
 *      { // finger
 *          handId
 *          id
 *          timeVisible
 *          tipPosition
 *          direction
 *          width
 *          length
 *          +extended (bool)
 *          type
 *          +carpPosition
 *          +mcpPosition
 *          +pipPosition
 *          +dipPosition
 *          +btipPosition
 *          +bases []
 *      }
 *  ]
 * }
 */
public class LeapWebsocketServer : MonoBehaviour {
    [SerializeField]
    LeapServiceProvider _leapServiceProvider;

    WebSocketServer _server;
    List<LeapBehavior> _leapBehaviors = new List<LeapBehavior>();

    class Vector3Converter : JsonConverter<Vector3> {
        public override void WriteJson(JsonWriter writer, Vector3 v, JsonSerializer serializer) {
            writer.WriteValue(new float[] {v.x, v.y, v.z});
        }

        public override Vector3 ReadJson(JsonReader reader, Type objectType, Vector3 existingValue, bool hasExistingValue, JsonSerializer serializer) {
            throw new NotImplementedException();
        }
    }
    class QuaternionConverter : JsonConverter<Quaternion> {
        public override void WriteJson(JsonWriter writer, Quaternion q, JsonSerializer serializer) {
            writer.WriteValue(new float[] {q.x, q.y, q.z, q.w});
        }

        public override Quaternion ReadJson(JsonReader reader, Type objectType, Quaternion existingValue, bool hasExistingValue, JsonSerializer serializer) {
            throw new NotImplementedException();
        }
    }
    class BasisConverter : JsonConverter<LeapTransform> {
        public override void WriteJson(JsonWriter writer, LeapTransform basis, JsonSerializer serializer) {
            writer.WriteValue(new float[][] { // TODO check line/cols
                new float[] {basis.xBasis.x, basis.xBasis.y, basis.xBasis.z},
                new float[] {basis.yBasis.x, basis.yBasis.y, basis.yBasis.z},
                new float[] {basis.zBasis.x, basis.zBasis.y, basis.zBasis.z}
            });
        }

        public override LeapTransform ReadJson(JsonReader reader, Type objectType, LeapTransform existingValue, bool hasExistingValue, JsonSerializer serializer) {
            throw new NotImplementedException();
        }
    }
    
    // https://github.com/HyroVitalyProtago/LeapMotionRemoteUnity/blob/main/RemoteController.cs
    // https://www.newtonsoft.com/json/help/html/ContractResolver.htm
    public class ConverterContractResolver : CamelCasePropertyNamesContractResolver {
        public new static readonly ConverterContractResolver Instance = new ConverterContractResolver();

        protected override JsonProperty CreateProperty(MemberInfo member, MemberSerialization memberSerialization) {
            JsonProperty property = base.CreateProperty(member, memberSerialization);

            // TODO
            // all x,y,z (,w) into an array (also in basis)
            // - remove normalized,magnitude,sqrMagnitude for Vectors
            // - remove eulerAngles for Quaternions
            // - remove translation, rotation, scale for Basis

            if (property.PropertyName == "currentFramesPerSecond") {
                property.PropertyName = "currentFrameRate";
            }

            return property;
        }
    }
    
    public class LeapBehavior : WebSocketBehavior {
        private Action<LeapBehavior> _onClose;
        JsonSerializerSettings _serializerSettings = new() {
            Formatting = Formatting.Indented,
            ReferenceLoopHandling = ReferenceLoopHandling.Ignore,
            ContractResolver = ConverterContractResolver.Instance,
            Converters = new List<JsonConverter>() {
                new Vector3Converter(),
                new QuaternionConverter(),
                new BasisConverter()
            }
        };
        public LeapBehavior(Action<LeapBehavior> onClose) : base() {
            _onClose = onClose;
        }
        
        protected override void OnOpen() {
            Send("{\"serviceVersion\":\"2.3.1 + 33747\", \"version\":6}");
        }

        public void SendFrame(Frame frame) {
            if (State != WebSocketState.Open) return;
            
            string json = JsonConvert.SerializeObject(frame, _serializerSettings);
            
            // For each hands, extract fingers into pointables
            // For each finger, add properties extended, carpPosition, mcpPosition, pipPosition, dipPosition, btipPosition, bases

            if (frame.Hands.Count > 0) {
                Debug.Log(json);    
            }
            
            //Send(json);
        }

        protected override void OnMessage(MessageEventArgs e) {
            Debug.Log(e.Data);
        }

        protected override void OnClose(CloseEventArgs e) {
            _onClose(this);
        }
    }

    void Awake() {
        //_leapServiceProvider = Find<LeapServiceProvider>();

        _server = new WebSocketServer(6437);
        _server.AddWebSocketService<LeapBehavior>("/v6.json", () => {
            LeapBehavior lb = new LeapBehavior((o) => {
                lock (_leapBehaviors) {
                    _leapBehaviors.Remove(o);
                }
            });
            lock (_leapBehaviors) {
                _leapBehaviors.Add(lb);
            }
            
            return lb;
        });
        _server.Start();
        
        _leapServiceProvider.OnUpdateFrame += LeapServiceProviderOnOnUpdateFrame;
    }

    void LeapServiceProviderOnOnUpdateFrame(Frame frame) {
        lock (_leapBehaviors) {
            foreach (LeapBehavior leapBehavior in _leapBehaviors) {
                leapBehavior.SendFrame(frame);   
            }
        }
    }

    void Destroy() {
        _server.Stop();
    }
}
