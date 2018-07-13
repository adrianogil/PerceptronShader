Shader "Custom/Batman"
{
    Properties
    {
        _BatmanColor("Batman Color", Color) = (1,1,1,1)
        _BackgroundColor("Background Color", Color) = (0,0,0,1)
    }
    Subshader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            sampler _MainTex;

            float4 _BatmanColor;
            float4 _BackgroundColor;

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
                float2 uv = o.uv - float2(0.5, 0.5);
                uv.x = uv.x * 14.0 * 2;
                uv.y = uv.y * 10.0 * 2;

                float x = uv.x;

                float bat_y_1 = -1.0 * abs(abs(x) - 1.0) * abs(3.0 - abs(x))/ ((abs(x)-1.0)*(3.0-abs(x)));
                float bat_y_2 = 1.0 - pow(x/7.0, 2.0);
                float bat_y_3 = 2.0 * sqrt(bat_y_1) * (1.0+abs(abs(x)-3.0)/(abs(x)-3.0));

                float bat_y_func_1 = bat_y_3 * sqrt(bat_y_2) + (5.0+0.97 * (abs(x-0.5)+abs(x+0.5)) - 3 * (abs(x-0.75)+abs(x+0.75))) * (1+abs(1-abs(x))/(1-abs(x)));
                float bat_y_func_2 = -3 * sqrt(1-pow(x/7.0,2.0)) * sqrt(abs(abs(x)-4)/(abs(x)-4));
                float bat_y_func_3 = abs(x/2.0) - 0.0913722* pow(x,2.0) - 3.0 + sqrt(1-pow(abs(abs(x)-2)-1, 2.0));
                float bat_y_func_4 =  (2.71052 + (1.5 - 0.5 * abs(x))-1.35526 * sqrt(4 - pow(abs(x)-1, 2.0))) * sqrt(abs(abs(x)-1)/(abs(x)-1))+0.9;

                if (abs(bat_y_func_1 - uv.y) < 0.09 || abs(bat_y_func_2 - uv.y) < 0.09 || 
                    abs(bat_y_func_3 - uv.y) < 0.09 || abs(bat_y_func_4 - uv.y) < 0.09)
                {
                    return _BatmanColor;
                }


                return _BackgroundColor;
            }

            ENDCG
        }
    }
}