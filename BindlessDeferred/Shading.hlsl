//=================================================================================================
//
//  Bindless Deferred Texturing Sample
//  by MJP
//  http://mynameismjp.wordpress.com/
//
//  All code and content licensed under the MIT license
//
//=================================================================================================

// Options
#define ShadowMapMode_ ShadowMapMode_DepthMap_

#ifndef UseImplicitShadowDerivatives_
    #define UseImplicitShadowDerivatives_ 0
#endif

#define UseReceiverPlaneBias_ 1

// Set this to zero to make compile times quicker
#define UseGatherPCF_ 1

#include <DescriptorTables.hlsl>
#include <SH.hlsl>
#include <Shadows.hlsl>
#include <BRDF.hlsl>
#include <Quaternion.hlsl>
#include "AppSettings.hlsl"
#include "SharedTypes.h"

struct ShadingConstants
{
    float3 SunDirectionWS;
    float CosSunAngularRadius;
    float3 SunIrradiance;
    float SinSunAngularRadius;
    float3 CameraPosWS;

    float3 CursorDecalPos;
    float CursorDecalIntensity;
    Quaternion CursorDecalOrientation;
    float3 CursorDecalSize;
    uint CursorDecalTexIdx;
    uint NumXTiles;
    uint NumXYTiles;
    float NearClip;
    float FarClip;

    SH9Color SkySH;
};

struct LightConstants
{
    SpotLight Lights[MaxSpotLights];
    float4x4 ShadowMatrices[MaxSpotLights];
};

struct ShadingInput
{
    uint2 PositionSS;
    float3 PositionWS;
    float3 PositionWS_DX;
    float3 PositionWS_DY;
    float DepthVS;
    float3x3 TangentFrame;

    float4 AlbedoMap;
    float2 NormalMap;
    float RoughnessMap;
    float MetallicMap;

    StructuredBuffer<Decal> DecalBuffer;
    ByteAddressBuffer DecalClusterBuffer;
    ByteAddressBuffer SpotLightClusterBuffer;

    SamplerState AnisoSampler;

    ShadingConstants ShadingCBuffer;
    SunShadowConstants ShadowCBuffer;
    LightConstants LightCBuffer;
};

// distanceSquared and traceScreenSpaceRay1 functions taken from Morgan McGuire's SSR sample code: http://casual-effects.blogspot.com/2014/08/screen-space-ray-tracing.html

// We use G3D Innovation Engine's header files and preprocessor for loops. You can
// manually expand this if your framework does not have a similar
// feature, or simply eliminate all of the multi-layer code if you're
// working with a typical single-layer depth buffer.

float distanceSquared(float2 A, float2 B)
{
    A -= B;
    return dot(A, A);
}

/**
    \param csOrigin Camera-space ray origin, which must be 
    within the view volume and must have z < -0.01 and project within the valid screen rectangle

    \param csDirection Unit length camera-space ray direction

    \param projectToPixelMatrix A projection matrix that maps to pixel coordinates (not [-1, +1] normalized device coordinates)

    \param csZBuffer The depth or camera-space Z buffer, depending on the value of \a csZBufferIsHyperbolic

    \param csZBufferSize Dimensions of csZBuffer

    \param csZThickness Camera space thickness to ascribe to each pixel in the depth buffer
    
    \param csZBufferIsHyperbolic True if csZBuffer is an OpenGL depth buffer, false (faster) if
     csZBuffer contains (negative) "linear" camera space z values. Const so that the compiler can evaluate the branch based on it at compile time

    \param clipInfo See G3D::Camera documentation

    \param nearPlaneZ Negative number

    \param stride Step in horizontal or vertical pixels between samples. This is a float
     because integer math is slow on GPUs, but should be set to an integer >= 1

    \param jitterFraction  Number between 0 and 1 for how far to bump the ray in stride units
      to conceal banding artifacts

    \param maxSteps Maximum number of iterations. Higher gives better images but may be slow

    \param maxRayTraceDistance Maximum camera-space distance to trace before returning a miss

    \param hitPixel Pixel coordinates of the first intersection with the scene

    \param csHitPoint Camera space location of the ray hit

    Single-layer

 */
struct SSRParams
{
    float3 csOrigin;
    float3 csDirection;
    float4x4 projectToPixelMatrix;

    Texture2D<float> csZBuffer;
    float2 csZBufferSize;
    float csZThickness;
    //bool   			csZBufferIsHyperbolic;
    //float3 clipInfo;
    float nearPlaneZ;
    float stride;
    float jitterFraction;
    float maxSteps;
    float maxRayTraceDistance;
    float2 hitPixel;
    //out int which,
	float3 csHitPoint;
};

bool traceScreenSpaceRay1(inout SSRParams ssrParams)
{
    
    // Clip ray to a near plane in 3D (doesn't have to be *the* near plane, although that would be a good idea)
    float rayLength = ((ssrParams.csOrigin.z + ssrParams.csDirection.z * ssrParams.maxRayTraceDistance) > ssrParams.nearPlaneZ) ?
                        (ssrParams.nearPlaneZ - ssrParams.csOrigin.z) / ssrParams.csDirection.z :
                        ssrParams.maxRayTraceDistance;
    float3 csEndPoint = ssrParams.csDirection * rayLength + ssrParams.csOrigin;

    // Project into screen space
    float4 H0 = mul(float4(ssrParams.csOrigin, 1.0), ssrParams.projectToPixelMatrix);
    float4 H1 = mul(float4(csEndPoint, 1.0), ssrParams.projectToPixelMatrix);

    // There are a lot of divisions by w that can be turned into multiplications
    // at some minor precision loss...and we need to interpolate these 1/w values
    // anyway.
    //
    // Because the caller was required to clip to the near plane,
    // this homogeneous division (projecting from 4D to 2D) is guaranteed 
    // to succeed. 
    float k0 = 1.0 / H0.w;
    float k1 = 1.0 / H1.w;

    // Switch the original points to values that interpolate linearly in 2D
    float3 Q0 = ssrParams.csOrigin * k0;
    float3 Q1 = csEndPoint * k1;

	// Screen-space endpoints
    float2 P0 = H0.xy * k0;
    float2 P1 = H1.xy * k1;

    // [Optional clipping to frustum sides here]

    // Initialize to off screen
    ssrParams.hitPixel = float2(-1.0, -1.0);
    //which = 0; // Only one layer

    // If the line is degenerate, make it cover at least one pixel
    // to avoid handling zero-pixel extent as a special case later
    float P1x = (distanceSquared(P0, P1) < 0.0001) ? 0.01 : 0.0;
    P1 += P1x.xx;

    float2 delta = P1 - P0;

    // Permute so that the primary iteration is in x to reduce
    // large branches later
    bool permute = false;
    if (abs(delta.x) < abs(delta.y))
    {
		// More-vertical line. Create a permutation that swaps x and y in the output
        permute = true;

        // Directly swizzle the inputs
        delta = delta.yx;
        P1 = P1.yx;
        P0 = P0.yx;
    }
    
	// From now on, "x" is the primary iteration direction and "y" is the secondary one

    float stepDirection = sign(delta.x);
    float invdx = stepDirection / delta.x;
    float2 dP = float2(stepDirection, invdx * delta.y);

    // Track the derivatives of Q and k
    float3 dQ = (Q1 - Q0) * invdx;
    float dk = (k1 - k0) * invdx;

    // Scale derivatives by the desired pixel stride
    dP *= ssrParams.stride;
    dQ *= ssrParams.stride;
    dk *= ssrParams.stride;

    // Offset the starting values by the jitter fraction
    P0 += dP * ssrParams.jitterFraction;
    Q0 += dQ * ssrParams.jitterFraction;
    k0 += dk * ssrParams.jitterFraction;

	// Slide P from P0 to P1, (now-homogeneous) Q from Q0 to Q1, and k from k0 to k1
    float3 Q = Q0;
    float k = k0;

	// We track the ray depth at +/- 1/2 pixel to treat pixels as clip-space solid 
	// voxels. Because the depth at -1/2 for a given pixel will be the same as at 
	// +1/2 for the previous iteration, we actually only have to compute one value 
	// per iteration.
    float prevZMaxEstimate = ssrParams.csOrigin.z;
    float stepCount = 0.0;
    float rayZMax = prevZMaxEstimate, rayZMin = prevZMaxEstimate;
    float sceneZMax = rayZMax + 1e4;

    // P1.x is never modified after this point, so pre-scale it by 
    // the step direction for a signed comparison
    float end = P1.x * stepDirection;

    // We only advance the z field of Q in the inner loop, since
    // Q.xy is never used until after the loop terminates.

    for (float2 P = P0;
        ((P.x * stepDirection) <= end) &&
        (stepCount < ssrParams.maxSteps) &&
        ((rayZMax < sceneZMax - ssrParams.csZThickness) ||
            (rayZMin > sceneZMax)) &&
        (sceneZMax != 0.0);
        P += dP, Q.z += dQ.z, k += dk, stepCount += 1.0)
    {
                
        ssrParams.hitPixel = permute ? P.yx : P;

        // The depth range that the ray covers within this loop
        // iteration.  Assume that the ray is moving in increasing z
        // and swap if backwards.  Because one end of the interval is
        // shared between adjacent iterations, we track the previous
        // value and then swap as needed to ensure correct ordering
        rayZMin = prevZMaxEstimate;

        // Compute the value at 1/2 pixel into the future
        rayZMax = (dQ.z * 0.5 + Q.z) / (dk * 0.5 + k);
        prevZMaxEstimate = rayZMax;
        if (rayZMin > rayZMax)
        {
            //swap(rayZMin, rayZMax);
            float t = rayZMin;
            rayZMin = rayZMax;
            rayZMax = t;
        }

        // Camera-space z of the background
        sceneZMax = ssrParams.csZBuffer[uint2(ssrParams.hitPixel)].r;
        //sceneZMax = texelFetch(ssrParams.csZBuffer, int2(ssrParams.hitPixel), 0).r;

        // This compiles away when csZBufferIsHyperbolic = false
        /*if (csZBufferIsHyperbolic) {
            sceneZMax = reconstructCSZ(sceneZMax, clipInfo);
        }*/
    } // pixel on ray

    Q.xy += dQ.xy * stepCount;
    ssrParams.csHitPoint = Q * (1.0 / k);

    // Matches the new loop condition:
    return (rayZMax >= sceneZMax - ssrParams.csZThickness) && (rayZMin <= sceneZMax);
}

//-------------------------------------------------------------------------------------------------
// Calculates the lighting result for an analytical light source
//-------------------------------------------------------------------------------------------------
float3 CalcLighting(in float3 normal, in float3 lightDir, in float3 peakIrradiance,
                    in float3 diffuseAlbedo, in float3 specularAlbedo, in float roughness,
                    in float3 positionWS, in float3 cameraPosWS)
{
    float3 lighting = diffuseAlbedo * (1.0f / 3.14159f);

    float3 view = normalize(cameraPosWS - positionWS);
    const float nDotL = saturate(dot(normal, lightDir));
    if(nDotL > 0.0f)
    {
        const float nDotV = saturate(dot(normal, view));
        float3 h = normalize(view + lightDir);

        float3 fresnel = Fresnel(specularAlbedo, h, lightDir);

        float specular = GGX_Specular(roughness, normal, h, view, lightDir);
        lighting += specular * fresnel;
    }

    return lighting * nDotL * peakIrradiance;
}

//-------------------------------------------------------------------------------------------------
// Calculates the full shading result for a single pixel. Note: some of the input textures
// are passed directly to this function instead of through the ShadingInput struct in order to
// work around incorrect behavior from the shader compiler
//-------------------------------------------------------------------------------------------------
float3 ShadePixel(in ShadingInput input, in Texture2DArray SunShadowMap,
                  in Texture2DArray SpotLightShadowMap, in SamplerComparisonState ShadowSampler)
{
    float3 vtxNormalWS = input.TangentFrame._m20_m21_m22;
    float3 normalWS = vtxNormalWS;
    float3 positionWS = input.PositionWS;

    const ShadingConstants CBuffer = input.ShadingCBuffer;
    const SunShadowConstants ShadowCBuffer = input.ShadowCBuffer;

    float3 viewWS = normalize(CBuffer.CameraPosWS - positionWS);

    if(AppSettings.EnableNormalMaps)
    {
        // Sample the normal map, and convert the normal to world space
        float3 normalTS;
        normalTS.xy = input.NormalMap * 2.0f - 1.0f;
        normalTS.z = sqrt(1.0f - saturate(normalTS.x * normalTS.x + normalTS.y * normalTS.y));
        normalWS = normalize(mul(normalTS, input.TangentFrame));
    }

    float4 albedoMap =  1.0f;
    if(AppSettings.EnableAlbedoMaps)
        albedoMap = input.AlbedoMap;

    float metallic = saturate(input.MetallicMap);
    float3 diffuseAlbedo = lerp(albedoMap.xyz, 0.0f, metallic);
    float3 specularAlbedo = lerp(0.03f, albedoMap.xyz, metallic) * (AppSettings.EnableSpecular ? 1.0f : 0.0f);

    float roughnessMap = input.RoughnessMap;
    float roughness = roughnessMap * roughnessMap;

    float depthVS = input.DepthVS;

    // Compute shared cluster lookup data
    uint2 pixelPos = uint2(input.PositionSS);
    float zRange = CBuffer.FarClip - CBuffer.NearClip;
    float normalizedZ = saturate((depthVS - CBuffer.NearClip) / zRange);
    uint zTile = normalizedZ * NumZTiles;

    uint3 tileCoords = uint3(pixelPos / ClusterTileSize, zTile);
    uint clusterIdx = (tileCoords.z * CBuffer.NumXYTiles) + (tileCoords.y * CBuffer.NumXTiles) + tileCoords.x;

    float3 positionNeighborX = input.PositionWS + input.PositionWS_DX;
    float3 positionNeighborY = input.PositionWS + input.PositionWS_DY;

    // Apply decals
    uint numDecals = 0;
    if(AppSettings.RenderDecals)
    {
        uint clusterOffset = clusterIdx * DecalElementsPerCluster;

        // Loop over the number of 4-byte elements needed for each cluster
        [unroll]
        for(uint elemIdx = 0; elemIdx < DecalElementsPerCluster; ++elemIdx)
        {
            // Loop until we've processed every raised bit
            uint clusterElemMask = input.DecalClusterBuffer.Load((clusterOffset + elemIdx) * 4);
            while(clusterElemMask)
            {
                uint bitIdx = firstbitlow(clusterElemMask);
                clusterElemMask &= ~(1 << bitIdx);
                uint decalIdx = bitIdx + (elemIdx * 32);
                Decal decal = input.DecalBuffer[decalIdx];
                float3x3 decalRot = QuatTo3x3(decal.Orientation);

                // Apply the decal projection, and branch over the decal if we're outside of its bounding box
                float3 localPos = positionWS - decal.Position;
                localPos = mul(localPos, transpose(decalRot));
                float3 decalUVW = localPos / decal.Size;
                decalUVW.y *= -1;
                if(decalUVW.x >= -1.0f && decalUVW.x <= 1.0f &&
                   decalUVW.y >= -1.0f && decalUVW.y <= 1.0f &&
                   decalUVW.z >= -1.0f && decalUVW.z <= 1.0f)
                {
                    // Pull out the right textures from the descriptor array
                    float2 decalUV = saturate(decalUVW.xy * 0.5f + 0.5f);
                    Texture2D decalAlbedoMap = Tex2DTable[NonUniformResourceIndex(decal.AlbedoTexIdx)];
                    Texture2D decalNormalMap = Tex2DTable[NonUniformResourceIndex(decal.NormalTexIdx)];

                    // Calculate decal UV gradients
                    float3 decalPosNeighborX = positionNeighborX - decal.Position;
                    decalPosNeighborX = mul(decalPosNeighborX, transpose(decalRot));
                    decalPosNeighborX = decalPosNeighborX / decal.Size;
                    decalPosNeighborX.y *= -1;
                    float2 uvDX = saturate(decalPosNeighborX.xy * 0.5f + 0.5f) - decalUV;

                    float3 decalPosNeighborY = positionNeighborY - decal.Position;
                    decalPosNeighborY = mul(decalPosNeighborY, transpose(decalRot));
                    decalPosNeighborY = decalPosNeighborY / decal.Size;
                    decalPosNeighborY.y *= -1;
                    float2 uvDY = saturate(decalPosNeighborY.xy * 0.5f + 0.5f) - decalUV;

                    float4 decalAlbedo = decalAlbedoMap.SampleGrad(input.AnisoSampler, decalUV, uvDX, uvDY);
                    float3 decalNormalTS = decalNormalMap.SampleGrad(input.AnisoSampler, decalUV, uvDX, uvDY).xyz;

                    float decalBlend = decalAlbedo.w;
                    // decalBlend *= saturate(dot(decalRot._m20_m21_m22, -vtxNormalWS) * 100.0f - 99.0f);

                    decalNormalTS = decalNormalTS * 2.0f - 1.0f;
                    decalNormalTS.z *= -1.0f;
                    float3 decalNormalWS = mul(decalNormalTS, decalRot);

                    // Blend the decal properties with the material properties
                    diffuseAlbedo = lerp(diffuseAlbedo, decalAlbedo.xyz, decalBlend);
                    normalWS = lerp(normalWS, decalNormalWS, decalBlend);
                }

                ++numDecals;
            }
        }
    }

    // Apply the decal "cursor", indicating where a new decal will be placed
    if(CBuffer.CursorDecalIntensity > 0.0f && CBuffer.CursorDecalTexIdx != uint(-1))
    {
        float3x3 decalRot = QuatTo3x3(CBuffer.CursorDecalOrientation);
        float3 localPos = positionWS - CBuffer.CursorDecalPos;
        localPos = mul(localPos, transpose(decalRot));
        float3 decalUVW = localPos / CBuffer.CursorDecalSize;
        decalUVW.y *= -1.0f;
        if(decalUVW.x >= -1.0f && decalUVW.x <= 1.0f &&
           decalUVW.y >= -1.0f && decalUVW.y <= 1.0f &&
           decalUVW.z >= -1.0f && decalUVW.z <= 1.0f)
        {
            float2 decalUV = saturate(decalUVW.xy * 0.5f + 0.5f);
            Texture2D<float4> decalAlbedoMap = Tex2DTable[CBuffer.CursorDecalTexIdx];
            float4 decalAlbedo = decalAlbedoMap.SampleLevel(input.AnisoSampler, decalUV, 0.0f);

            float decalBlend = decalAlbedo.w;
            // decalBlend *= saturate(dot(decalRot._m20_m21_m22, -vtxNormalWS) * 100.0f - 99.0f);

            diffuseAlbedo = lerp(diffuseAlbedo, decalAlbedo.xyz, decalBlend * CBuffer.CursorDecalIntensity);
        }
    }

    // Add in the primary directional light
    float3 output = 0.0f;

    if(AppSettings.EnableSun)
    {
        #if UseImplicitShadowDerivatives_
            // Forward path
            float sunShadowVisibility = SunShadowVisibility(positionWS, depthVS, SunShadowMap, ShadowSampler, ShadowCBuffer, 0);
        #else
            // Deferred path
            float sunShadowVisibility = SunShadowVisibility(positionWS, positionNeighborX, positionNeighborY,
                                                            depthVS, SunShadowMap, ShadowSampler, ShadowCBuffer, 0);
        #endif

        float3 sunDirection = CBuffer.SunDirectionWS;
        if(AppSettings.SunAreaLightApproximation)
        {
            float3 D = CBuffer.SunDirectionWS;
            float3 R = reflect(-viewWS, normalWS);
            float r = CBuffer.SinSunAngularRadius;
            float d = CBuffer.CosSunAngularRadius;
            float3 DDotR = dot(D, R);
            float3 S = R - DDotR * D;
            sunDirection = DDotR < d ? normalize(d * D + normalize(S) * r) : R;
        }
        output += CalcLighting(normalWS, sunDirection, CBuffer.SunIrradiance, diffuseAlbedo, specularAlbedo,
                                 roughness, positionWS, CBuffer.CameraPosWS) * sunShadowVisibility;
    }

    // Apply the spot lights
    uint numLights = 0;
    if(AppSettings.RenderLights)
    {
        uint clusterOffset = clusterIdx * SpotLightElementsPerCluster;

        // Loop over the number of 4-byte elements needed for each cluster
        [unroll]
        for(uint elemIdx = 0; elemIdx < SpotLightElementsPerCluster; ++elemIdx)
        {
            // Loop until we've processed every raised bit
            uint clusterElemMask = input.SpotLightClusterBuffer.Load((clusterOffset + elemIdx) * 4);
            while(clusterElemMask)
            {
                uint bitIdx = firstbitlow(clusterElemMask);
                clusterElemMask &= ~(1 << bitIdx);
                uint spotLightIdx = bitIdx + (elemIdx * 32);
                SpotLight spotLight = input.LightCBuffer.Lights[spotLightIdx];

                float3 surfaceToLight = spotLight.Position - positionWS;
                float distanceToLight = length(surfaceToLight);
                surfaceToLight /= distanceToLight;
                float angleFactor = saturate(dot(surfaceToLight, spotLight.Direction));
                float angularAttenuation = smoothstep(spotLight.AngularAttenuationY, spotLight.AngularAttenuationX, angleFactor);

                if(angularAttenuation > 0.0f)
                {
                    float d = distanceToLight / spotLight.Range;
                    float falloff = saturate(1.0f - (d * d * d * d));
                    falloff = (falloff * falloff) / (distanceToLight * distanceToLight + 1.0f);
                    float3 intensity = spotLight.Intensity * angularAttenuation * falloff;

                    // We have to use explicit gradients for spotlight shadows, since the looping/branching is non-uniform
                    float spotLightVisibility = SpotLightShadowVisibility(positionWS, positionNeighborX, positionNeighborY,
                                                                          input.LightCBuffer.ShadowMatrices[spotLightIdx],
                                                                          spotLightIdx, SpotLightShadowMap, ShadowSampler, 0.0f, 0);

                    output += CalcLighting(normalWS, surfaceToLight, intensity, diffuseAlbedo, specularAlbedo,
                                           roughness, positionWS, CBuffer.CameraPosWS) * spotLightVisibility;
                }

                ++numLights;
            }
        }
    }

    float3 ambient = EvalSH9Irradiance(normalWS, CBuffer.SkySH) * InvPi;
    ambient *= 0.1f; // Darken the ambient since we don't have any sky occlusion
    output += ambient * diffuseAlbedo;

    if(AppSettings.ShowLightCounts)
        output = lerp(output, float3(2.5f, 0.0f, 0.0f), numLights / 10.0f);

    if(AppSettings.ShowDecalCounts)
        output = lerp(output, float3(0.0f, 2.5f, 0.0f), numDecals / 10.0f);

    output = clamp(output, 0.0f, FP16Max);

    // Roughness test.
    if (AppSettings.EnableSSR &&
        ( input.PositionWS.y > 0 ||                                                       // Restricting the reflection to only the floor.
          ( ( AppSettings.SSRoughnessMinThreshold < AppSettings.SSRoughnessMaxThreshold ) &&  // Min threshold must always be lesser than max threshold
            ( roughness < AppSettings.SSRoughnessMinThreshold ||
              roughness > AppSettings.SSRoughnessMaxThreshold ) ) ) )
    {
        output = float3(0, 0, 0);
    }

    return output;
}