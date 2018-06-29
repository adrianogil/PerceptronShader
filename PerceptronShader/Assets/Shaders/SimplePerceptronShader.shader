Shader "Perceptron/Simple"
{
    Properties
    {
        _W0("W0", FLOAT) = 1
        _W1("W1", FLOAT) = 0
    }
    Subshader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            float _W0, _W1;

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
                if (dot(normalize(o.uv - float2(0.5, 0.5)), normalize(float2(_W0, _W1))) >= 0)
                {
                    return half4(0,0,1,1);
                }

                return half4(1,0,0,1);
            }

            ENDCG
        }
    }
}