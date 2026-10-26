// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::sync::LazyLock;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::ForwardsUOffset;
use flatbuffers::InvalidFlatbuffer;
use flatbuffers::Vector;
use flatbuffers::Verifiable;
use flatbuffers::Verifier;
use flatbuffers::VerifierOptions;
use flatbuffers::WIPOffset;
use vortex_array::dtype::DType;
use vortex_array::flatbuffers::FlatBuffer;
use vortex_array::flatbuffers::FlatBufferRoot;
use vortex_array::flatbuffers::WriteFlatBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::DynLayout;
use crate::LayoutBuildContext;
use crate::LayoutContext;
use crate::LayoutRef;
use crate::children::ViewedLayoutChildren;
use crate::flatbuffers::layout;
use crate::layouts::foreign::new_foreign_layout;
use crate::segments::SegmentId;
use crate::session::LayoutSessionExt;

pub(crate) static LAYOUT_VERIFIER: LazyLock<VerifierOptions> = LazyLock::new(|| {
    VerifierOptions {
        // Overridden
        max_tables: env::var("VORTEX_MAX_LAYOUT_TABLES")
            .ok()
            .and_then(|lmt| lmt.parse::<usize>().ok())
            .unwrap_or(1000000),
        max_depth: env::var("VORTEX_MAX_LAYOUT_DEPTH")
            .ok()
            .and_then(|lmt| lmt.parse::<usize>().ok())
            .unwrap_or(64),
        // Defaults from flatbuffers
        max_apparent_size: 1 << 31,
        ignore_missing_null_terminator: false,
    }
});

/// Verifies one layout table: its own fields, and the `encoding` and `row_count` of each direct
/// child, which a parent reads without building the child. The remaining fields of a child are
/// verified when that child is built, so only the parts of the tree a reader touches are verified.
struct LayoutNode;

impl Verifiable for LayoutNode {
    fn run_verifier(v: &mut Verifier, pos: usize) -> Result<(), InvalidFlatbuffer> {
        v.visit_table(pos)?
            .visit_field::<u16>("encoding", layout::Layout::VT_ENCODING, false)?
            .visit_field::<u64>("row_count", layout::Layout::VT_ROW_COUNT, false)?
            .visit_field::<ForwardsUOffset<Vector<'_, u8>>>(
                "metadata",
                layout::Layout::VT_METADATA,
                false,
            )?
            .visit_field::<ForwardsUOffset<Vector<'_, ForwardsUOffset<LayoutHeader>>>>(
                "children",
                layout::Layout::VT_CHILDREN,
                false,
            )?
            .visit_field::<ForwardsUOffset<Vector<'_, u32>>>(
                "segments",
                layout::Layout::VT_SEGMENTS,
                false,
            )?
            .finish();
        Ok(())
    }
}

/// The fields of a child layout that its parent reads without building it.
struct LayoutHeader;

impl Verifiable for LayoutHeader {
    fn run_verifier(v: &mut Verifier, pos: usize) -> Result<(), InvalidFlatbuffer> {
        v.visit_table(pos)?
            .visit_field::<u16>("encoding", layout::Layout::VT_ENCODING, false)?
            .visit_field::<u64>("row_count", layout::Layout::VT_ROW_COUNT, false)?
            .finish();
        Ok(())
    }
}

/// Verifies the layout table at `loc`, as described on [`LayoutNode`].
pub(crate) fn verify_layout_node(flatbuffer: &[u8], loc: usize) -> VortexResult<()> {
    LayoutNode::run_verifier(&mut Verifier::new(&LAYOUT_VERIFIER, flatbuffer), loc)?;
    Ok(())
}

/// Verifies the layout table at `loc` and every layout below it.
pub(crate) fn verify_layout_tree(flatbuffer: &[u8], loc: usize) -> VortexResult<()> {
    layout::Layout::run_verifier(&mut Verifier::new(&LAYOUT_VERIFIER, flatbuffer), loc)?;
    Ok(())
}

/// Parse a [`LayoutRef`] from a layout flatbuffer.
pub fn layout_from_flatbuffer(
    flatbuffer: FlatBuffer,
    dtype: &DType,
    layout_ctx: &ReadContext,
    ctx: &ReadContext,
    session: &VortexSession,
) -> VortexResult<LayoutRef> {
    layout_from_flatbuffer_with_options(flatbuffer, dtype, layout_ctx, ctx, session, false)
}

/// Parse a [`LayoutRef`] from a layout flatbuffer with unknown-encoding behavior control.
pub fn layout_from_flatbuffer_with_options(
    flatbuffer: FlatBuffer,
    dtype: &DType,
    layout_ctx: &ReadContext,
    ctx: &ReadContext,
    session: &VortexSession,
    allow_unknown: bool,
) -> VortexResult<LayoutRef> {
    let layout_session = session.layouts();
    let layouts = layout_session.registry();
    ForwardsUOffset::<LayoutNode>::run_verifier(
        &mut Verifier::new(&LAYOUT_VERIFIER, &flatbuffer),
        0,
    )?;
    // SAFETY: the root offset and the root table were verified above.
    let fb_layout = unsafe { flatbuffers::root_unchecked::<layout::Layout>(&flatbuffer) };
    let encoding_id = layout_ctx
        .resolve(fb_layout.encoding())
        .ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding()))?;
    let encoding = layouts.get(&encoding_id);

    if encoding.is_none() && allow_unknown {
        // A foreign layout is built eagerly from the whole subtree.
        verify_layout_tree(&flatbuffer, fb_layout._tab.loc())?;
        return foreign_layout_from_fb(fb_layout, dtype, layout_ctx);
    }
    let encoding =
        encoding.ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding()))?;

    // SAFETY: the root table was verified with `LayoutNode` above.
    let viewed_children = unsafe {
        ViewedLayoutChildren::new_unchecked(
            flatbuffer.clone(),
            fb_layout._tab.loc(),
            ctx.clone(),
            layout_ctx.clone(),
            layouts.clone(),
            allow_unknown,
            session.clone(),
        )
    };

    let build_ctx = LayoutBuildContext {
        session,
        array_read_ctx: ctx,
    };
    let layout = encoding.build(
        dtype,
        fb_layout.row_count(),
        fb_layout
            .metadata()
            .map(|m| m.bytes())
            .unwrap_or_else(|| &[]),
        fb_layout
            .segments()
            .unwrap_or_default()
            .iter()
            .map(SegmentId::from)
            .collect(),
        &viewed_children,
        &build_ctx,
    )?;

    Ok(layout)
}

fn foreign_layout_from_fb(
    fb_layout: layout::Layout<'_>,
    dtype: &DType,
    layout_ctx: &ReadContext,
) -> VortexResult<LayoutRef> {
    let encoding_id = layout_ctx
        .resolve(fb_layout.encoding())
        .ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding()))?;

    let children = fb_layout
        .children()
        .unwrap_or_default()
        .iter()
        .map(|child| foreign_layout_from_fb(child, dtype, layout_ctx))
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(new_foreign_layout(
        encoding_id,
        dtype.clone(),
        fb_layout.row_count(),
        fb_layout
            .metadata()
            .map(|m| m.bytes().to_vec())
            .unwrap_or_default(),
        fb_layout
            .segments()
            .unwrap_or_default()
            .iter()
            .map(SegmentId::from)
            .collect(),
        children,
    ))
}

impl dyn DynLayout + '_ {
    /// Serialize the layout into a [`FlatBufferBuilder`].
    pub fn flatbuffer_writer<'a>(
        &'a self,
        ctx: &'a LayoutContext,
    ) -> impl WriteFlatBuffer<Target<'a> = layout::Layout<'a>> + FlatBufferRoot + 'a {
        LayoutFlatBufferWriter { layout: self, ctx }
    }
}

/// An adapter struct for writing a layout to a FlatBuffer.
struct LayoutFlatBufferWriter<'a> {
    layout: &'a dyn DynLayout,
    ctx: &'a LayoutContext,
}

impl FlatBufferRoot for LayoutFlatBufferWriter<'_> {}

impl WriteFlatBuffer for LayoutFlatBufferWriter<'_> {
    type Target<'fb> = layout::Layout<'fb>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        // First we recurse into the children and write them out
        let child_layouts = self.layout.children()?;
        let children = child_layouts
            .iter()
            .map(|layout| {
                LayoutFlatBufferWriter {
                    layout: layout.as_ref(),
                    ctx: self.ctx,
                }
                .write_flatbuffer(fbb)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let children = (!children.is_empty()).then(|| fbb.create_vector(&children));

        // Next we write out the metadata if it's non-empty.
        let metadata = self.layout.metadata();
        let metadata = (!metadata.is_empty()).then(|| fbb.create_vector(&metadata));

        let segments = self
            .layout
            .segment_ids()
            .into_iter()
            .map(|s| *s)
            .collect::<Vec<_>>();
        let segments = (!segments.is_empty()).then(|| fbb.create_vector(&segments));

        // Dictionary-encode the layout ID
        let encoding = self.ctx.intern(&self.layout.encoding_id()).ok_or_else(|| {
            vortex_err!(
                "Layout encoding {} not permitted by ctx",
                self.layout.encoding_id()
            )
        })?;

        Ok(layout::Layout::create(
            fbb,
            &layout::LayoutArgs {
                encoding,
                row_count: self.layout.row_count(),
                metadata,
                children,
                segments,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use flatbuffers::FlatBufferBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::LAYOUT_VERIFIER;
    use super::layout_from_flatbuffer_with_options;
    use super::verify_layout_tree;
    use crate::LayoutEncodingId;
    use crate::flatbuffers::layout as fbl;
    use crate::session::LayoutSession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<LayoutSession>());

    fn finish(
        mut fbb: FlatBufferBuilder<'_>,
        root: flatbuffers::WIPOffset<fbl::Layout<'_>>,
    ) -> Vec<u8> {
        fbb.finish_minimal(root);
        let (buf, start) = fbb.collapse();
        buf[start..].to_vec()
    }

    fn aligned(bytes: Vec<u8>) -> vortex_array::flatbuffers::FlatBuffer {
        vortex_array::flatbuffers::FlatBuffer::align_from(vortex_buffer::ByteBuffer::from(bytes))
    }

    /// A child whose metadata vector is corrupt fails when it is built, not when its parent is.
    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    #[test]
    fn corrupt_child_fails_only_when_built() {
        const MARKER: [u8; 4] = [0xAB, 0xCD, 0xEF, 0x12];
        let mut fbb = FlatBufferBuilder::new();
        let children = [vec![1u8], MARKER.to_vec()].map(|metadata| {
            let metadata = fbb.create_vector(&metadata);
            fbl::Layout::create(
                &mut fbb,
                &fbl::LayoutArgs {
                    encoding: 1,
                    row_count: 3,
                    metadata: Some(metadata),
                    children: None,
                    segments: None,
                },
            )
        });
        let children = fbb.create_vector(&children);
        let root = fbl::Layout::create(
            &mut fbb,
            &fbl::LayoutArgs {
                encoding: 0,
                row_count: 3,
                metadata: None,
                children: Some(children),
                segments: None,
            },
        );
        let mut bytes = finish(fbb, root);
        let marker = bytes
            .windows(MARKER.len())
            .position(|window| window == MARKER)
            .unwrap();
        bytes[marker - 4..marker].copy_from_slice(&u32::MAX.to_le_bytes());
        let root_loc = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        assert!(verify_layout_tree(&bytes, root_loc).is_err());

        let i32 = DType::Primitive(PType::I32, Nullability::NonNullable);
        let dtype = DType::Struct(
            StructFields::from_iter([("a", i32.clone()), ("b", i32)]),
            Nullability::NonNullable,
        );
        let layout_ctx = ReadContext::new([
            LayoutEncodingId::new("vortex.struct"),
            LayoutEncodingId::new("vortex.test.foreign_child_layout"),
        ]);

        let layout = layout_from_flatbuffer_with_options(
            aligned(bytes),
            &dtype,
            &layout_ctx,
            &ReadContext::new([]),
            &SESSION,
            true,
        )
        .unwrap();

        assert_eq!(layout.slot(1).unwrap().unwrap().metadata(), vec![1]);
        assert!(layout.slot(2).is_err());
    }

    /// Nesting beyond the verifier's depth limit is rejected when the too-deep layout is built.
    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    #[test]
    fn layout_deeper_than_limit_fails_when_built() {
        let mut fbb = FlatBufferBuilder::new();
        let mut layout = fbl::Layout::create(&mut fbb, &fbl::LayoutArgs::default());
        for _ in 0..LAYOUT_VERIFIER.max_depth {
            let children = fbb.create_vector(&[layout]);
            layout = fbl::Layout::create(
                &mut fbb,
                &fbl::LayoutArgs {
                    children: Some(children),
                    ..Default::default()
                },
            );
        }
        let bytes = finish(fbb, layout);

        let mut layout = layout_from_flatbuffer_with_options(
            aligned(bytes),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            &ReadContext::new([LayoutEncodingId::new("vortex.chunked")]),
            &ReadContext::new([]),
            &SESSION,
            false,
        )
        .unwrap();
        for _ in 1..LAYOUT_VERIFIER.max_depth {
            layout = layout.slot(0).unwrap().unwrap();
        }
        let err = layout.slot(0).unwrap_err();
        assert!(err.to_string().contains("nested deeper"), "{err}");
    }

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    #[test]
    fn unknown_layout_encoding_allow_unknown() {
        let mut fbb = FlatBufferBuilder::new();

        let child_metadata = fbb.create_vector(&[9u8]);
        let child = fbl::Layout::create(
            &mut fbb,
            &fbl::LayoutArgs {
                encoding: 1,
                row_count: 3,
                metadata: Some(child_metadata),
                children: None,
                segments: None,
            },
        );

        let children = fbb.create_vector(&[child]);
        let metadata = fbb.create_vector(&[1u8, 2, 3]);
        let segments = fbb.create_vector(&[7u32]);
        let root = fbl::Layout::create(
            &mut fbb,
            &fbl::LayoutArgs {
                encoding: 0,
                row_count: 10,
                metadata: Some(metadata),
                children: Some(children),
                segments: Some(segments),
            },
        );
        fbb.finish_minimal(root);
        let (buf, start) = fbb.collapse();
        let layout_buffer = vortex_array::flatbuffers::FlatBuffer::align_from(
            vortex_buffer::ByteBuffer::from(buf).slice(start..),
        );

        let layout_ctx = ReadContext::new([
            LayoutEncodingId::new("vortex.test.foreign_layout"),
            LayoutEncodingId::new("vortex.test.foreign_child_layout"),
        ]);
        let array_ctx = ReadContext::new([]);

        let layout = layout_from_flatbuffer_with_options(
            layout_buffer,
            &DType::Variant(Nullability::Nullable),
            &layout_ctx,
            &array_ctx,
            &SESSION,
            true,
        )
        .unwrap();

        assert_eq!(layout.encoding_id().as_ref(), "vortex.test.foreign_layout");
        assert_eq!(layout.row_count(), 10);
        assert_eq!(layout.metadata(), vec![1, 2, 3]);
        assert_eq!(layout.segment_ids().len(), 1);
        assert_eq!(*layout.segment_ids()[0], 7);
        assert_eq!(layout.nchildren(), 1);

        let child = layout.slot(0).unwrap().unwrap();
        assert_eq!(
            child.encoding_id().as_ref(),
            "vortex.test.foreign_child_layout"
        );
        assert_eq!(child.metadata(), vec![9]);
    }
}
