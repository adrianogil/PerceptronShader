Shader "2DFunction/Linear"
{
    Properties
    {
        _FunctionRate("A", FLOAT) = 1
        _FunctionBias("B", FLOAT) = 0
    }
    Subshader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            float _FunctionRate;
            float _FunctionBias;

            struct vert_input
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct vert_output
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            vert_output vert(vert_input i)
            {
                vert_output o;

                o.vertex = UnityObjectToClipPos(i.vertex);
                o.uv = i.uv;

                return o;
            }

            half4 frag(vert_output o) : COLOR
            {
                if (o.uv.y < _FunctionRate*o.uv.x + _FunctionBias)
                {
                    return half4(0,0,1,1);
                }

                return half4(1,0,0,1);
            }

            ENDCG
        }
    }
}