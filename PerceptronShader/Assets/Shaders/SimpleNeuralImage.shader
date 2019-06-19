Shader "Perceptron/SimpleNeuralImage"
{
    Properties
    {
        _W0("W0", FLOAT) = 1
        _W1("W1", FLOAT) = 0
        _W2("W2", FLOAT) = 0
        _W3("W3", FLOAT) = 0
        _W4("W4", FLOAT) = 0
        _W5("W5", FLOAT) = 0
        _W6("W6", FLOAT) = 0
        _W7("W7", FLOAT) = 0
        _W8("W8", FLOAT) = 0
        _W9("W9", FLOAT) = 0

        _B0("B0", FLOAT) = 0
        _B1("B0", FLOAT) = 0
        _B2("B0", FLOAT) = 0
    }
    Subshader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            float _W0, _W1, _W2, _W3, _W4, _W5, _W6, _W7, _W8, _W9;
            float _B0, _B1, _B2;

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

            float tansig(float w1, float w2, float b, float p1, float p2)
            {
                float n = w1 * p1 + w2 * p2 + b;
                float e1 = exp(n);
                float e2 = exp(-n);
                return (e1 + e2) / (e1 + e2);

                // return 1;
            }


            float4 frag(vert_output o) : COLOR
            {
                float3 color = float3(1,0,0);

                color.r = 1.0 / (1.0 + exp(-(_W0 * o.uv.x + _W1 * o.uv.y + _B0)));
                color.g = 1.0 / (1.0 + exp(-(_W2 * o.uv.x + _W3 * o.uv.y + _B1)));
                color.b = 1.0 / (1.0 + exp(-(_W4 * o.uv.x + _W5 * o.uv.y + _B2)));
                // color.r = tansig(_W0, _W1, _B0, o.uv.x, o.uv.y);
                // color.g = tansig(_W2, _W3, _B1, o.uv.x, o.uv.y);
                // color.b = tansig(_W4, _W5, _B2, o.uv.x, o.uv.y);

                return float4(color, 1);
            }

            ENDCG
        }
    }
}